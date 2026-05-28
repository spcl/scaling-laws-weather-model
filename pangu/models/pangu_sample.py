import sys
sys.path.append("/data")
from era5_data import utils, utils_data
from era5_data.config import cfg
from torch import nn
import torch
import copy
from era5_data import score
import os
import numpy as np
import time
import gc
import glob
from torch.nn.parallel import DistributedDataParallel as DDP


def save_checkpoint(model, params, checkpoint_path, iters, epoch, optimizer, scheduler, train_dataloader, log_to_screen, logger):
    """
    Save out checkpoint
    """

    if log_to_screen:
        logger.info(f"Writing checkpoint to {checkpoint_path}")

    with torch.no_grad():
        # legacy mode
        # start timer
        store_start = time.time()
        checkpoint_fname = checkpoint_path
        store_dict = {
            "iters": iters,
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "params": params,
            "dataloader_state_dict": train_dataloader.state_dict(),
            "dataloader_sampler_state_dict": train_dataloader.sampler.state_dict() if hasattr(train_dataloader.sampler, 'state_dict') else None,
        }
        if scheduler is not None:
            store_dict["scheduler_state_dict"] = scheduler.state_dict()
        torch.save(store_dict, checkpoint_fname)
        # stop timer
        store_stop = time.time()

        # report time
        if log_to_screen:
            logger.info(f"Save checkpoint: {(store_stop - store_start):.2f} sec ({sys.getsizeof(store_dict)/(1024.**3)}) GB")

    # # optional let's clean up
    # gc.collect()
    # torch.cuda.empty_cache()

    # sync the device
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()


def restore_checkpoint(checkpoint_path, train_dataloader, model, optimizer, scheduler, logger,
                        load_optimizer=True, load_scheduler=True, load_counters=True,
                        load_dataloader=True, checkpoint_mode="flexible", log_to_screen=True, step=None):
    """
    Restore a checkpoint
    """

    # Support restoring from checkpoint files with val_loss and step in the filename
    # If checkpoint_path does not include '_valloss' and '_step', use glob to find the latest matching checkpoint file
    
    if ("_valloss" not in checkpoint_path) or ("_step" not in checkpoint_path):
        # Build glob pattern
        base, ext = os.path.splitext(checkpoint_path)
        if step is not None:
            pattern = f"{base}_valloss*_step{step}{ext}"
        else:
            pattern = f"{base}_valloss*_step*{ext}"
        matches = glob.glob(pattern)
        if matches:
            # Sort by step (extract from filename)
            def extract_step(fname):
                import re
                m = re.search(r"_step(\d+)", fname)
                return int(m.group(1)) if m else -1
            matches.sort(key=extract_step, reverse=True)  # latest step first
            checkpoint_path = matches[0]
            if log_to_screen:
                logger.info(f"Restoring checkpoint from: {checkpoint_path}")
        else:
            if log_to_screen:
                logger.warning(f"No checkpoint found matching pattern: {pattern}, restarting from scratch")
            return model, optimizer, scheduler, 0, train_dataloader
    else:
        checkpoint_path = checkpoint_path

    if log_to_screen:
        logger.info("Loading checkpoint %s in %s mode" % (checkpoint_path, checkpoint_mode))

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state"]
    if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, "module.")
    model.load_state_dict(state_dict, strict=True)
    
    if "dataloader_state_dict" in checkpoint and load_dataloader:
        train_dataloader.load_state_dict(checkpoint["dataloader_state_dict"])
        print("Loaded dataloader state dict")
        
    if "dataloader_sampler_state_dict" in checkpoint and load_dataloader:
        train_dataloader.sampler.load_state_dict(checkpoint["dataloader_sampler_state_dict"])
        print("Loaded dataloader sampler state dict")
        
    # If finetuning, restore checkpoint does not load optimizer state, instead uses config specified lr.
    if "optimizer_state_dict" in checkpoint and load_optimizer:
        print('load optimizer')
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if "scheduler_state_dict" in checkpoint and load_scheduler and (scheduler is not None):
        print('load scheduler')
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    iters = 0
    if "iters" in checkpoint and load_counters:
        iters = checkpoint["iters"]
        print('load counters, iter={}'.format(iters))


    # let's clean up
    gc.collect()
    torch.cuda.empty_cache()

    # sync the device
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()

    return model, optimizer, scheduler, iters, train_dataloader

def train(model, train_loader, val_loader, optimizer, lr_scheduler, res_path, device, rank, writer, logger, start_epoch,
          checkpoint_path=None, resume=False):
    '''Training code'''
    # Prepare for the optimizer and scheduler
    # lr_scheduler = torch.optim.scheduler.CosineAnnealingLR(optimizer, 10, eta_min=0, last_epoch=- 1, verbose=False) #used in the paper

    # Loss function
    criterion = nn.L1Loss(reduction='none')
    criterion_val = nn.MSELoss(reduction='none')

    # training epoch
    epochs = cfg.PG.TRAIN.EPOCHS

    loss_list = []
    best_loss = float('inf')
    epochs_since_last_improvement = 0
    best_model = None
    # scaler = torch.cuda.amp.GradScaler()

    # Load constants and teleconnection indices
    aux_constants = utils_data.loadAllConstants(
        device=device)  # 'weather_statistics','weather_statistics_last','constant_maps','tele_indices','variable_weights'
    upper_weights, surface_weights = aux_constants['variable_weights'] # use the same weights in training and validation

    iters = 0
    if resume:
        model, optimizer, lr_scheduler, iters, train_loader = restore_checkpoint(checkpoint_path, train_loader, model, optimizer, lr_scheduler, logger,
                        load_optimizer=True, load_scheduler=True, load_counters=True,
                        load_dataloader=True, checkpoint_mode="standard", log_to_screen=True)
    
    # print("model.device after restore", model.device)
    model = model.to(device)
    # print("model.device after to", model.device)
    model = DDP(model, device_ids=[rank])
    
    # Train a single Pangu-Weather model
    for i in range(start_epoch, epochs + 1):
        epoch_loss = 0.0

        for id, train_data in enumerate(train_loader):
            iters += 1
            # Load weather data at time t as the input; load weather data at time t+336 as the output
            # Note the data need to be randomly shuffled
            input, target = train_data
            # input = torch.tensor(input, dtype=torch.float32)
            # target = torch.tensor(target, dtype=torch.float32)
            input_surface = torch.tensor(input[:,:4,:,:], dtype=torch.float32)
            input = input[:,4:,:,:].view(1,-1,13,721,1440)
            # print("input shape 0", input.shape)
            target_surface = torch.tensor(target[:,:4,:,:], dtype=torch.float32)
            target = target[:,4:,:,:].view(1,-1,13,721,1440)
            # print("target shape 0", target.shape)
            input, target, input_surface, target_surface = input.to(device, dtype=torch.float32), \
                target.to(device, dtype=torch.float32), input_surface.to(device, dtype=torch.float32), \
                target_surface.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float32):
            # /with torch.cuda.amp.autocast():
                model.train()

                # Note the input and target need to be normalized (done within the function)
                # Call the model and get the output
                output, output_surface = model(input, input_surface, 
                                aux_constants['constant_maps'],
                                aux_constants['const_h'])  # (1,5,13,721,1440)

                
            # We use the MAE loss to train the model
            # Different weight can be applied for different fields if needed
            loss_surface = criterion(output_surface, target_surface)
            # print(loss_surface)
            weighted_surface_loss = torch.mean(loss_surface * surface_weights)

            loss_upper = criterion(output, target)
            # print(loss_upper)
            weighted_upper_loss = torch.mean(loss_upper * upper_weights)
            # The weight of surface loss is 0.25
            loss = weighted_upper_loss + weighted_surface_loss * 0.25
            
            print("Step {i} loss: {loss}".format(i=iters, loss=loss.item()))

            # Call the backward algorithm and calculate the gratitude of parameters
            # scaler.scale(loss).backward()
            loss.backward()
            # Update model parameters with Adam optimizer
            # scaler.step(optimizer)
            # scaler.update()
            optimizer.step()
            epoch_loss += loss.item()
            
            if iters % 5 == 0:
                
                # validation
                with torch.inference_mode():
                    with torch.no_grad():
                        val_count = 0
                        for val_data in val_loader:
                            val_count += 1
                            
                            input, target = val_data
                            input = torch.tensor(input, dtype=torch.float32)
                            target = torch.tensor(target, dtype=torch.float32)
                            input_surface = torch.tensor(input[:,:4,:,:], dtype=torch.float32)
                            input = input[:,4:,:,:].view(1,-1,13,721,1440)
                            # print("input shape 0", input.shape)
                            target_surface = torch.tensor(target[:,:4,:,:], dtype=torch.float32)
                            target = target[:,4:,:,:].view(1,-1,13,721,1440)
                            # print("target shape 0", target.shape)
                            input, target, input_surface, target_surface = input.to(device), \
                                target.to(device), input_surface.to(device), \
                                target_surface.to(device)

                            optimizer.zero_grad()
                            with torch.autocast(device_type='cuda', dtype=torch.float32):
                            # /with torch.cuda.amp.autocast():
                                model.eval()

                                # Note the input and target need to be normalized (done within the function)
                                # Call the model and get the output
                                output, output_surface = model(input, input_surface, 
                                                aux_constants['constant_maps'],
                                                aux_constants['const_h'])  # (1,5,13,721,1440)

                                
                            # We use the MAE loss to train the model
                            # Different weight can be applied for different fields if needed
                            val_loss_surface = criterion_val(output_surface, target_surface)
                            weighted_surface_loss = torch.sum(val_loss_surface * surface_weights)

                            val_loss_upper = criterion_val(output, target)
                            weighted_upper_loss = torch.sum(val_loss_upper * upper_weights)
                            output_z500 = output[0,0,7,:,:]  # 50, 100, 150, 200, 250, 300, 400, 500, 600, ...
                            target_z500 = target[0,0,7,:,:]
                            rmse_z500 = torch.sqrt(torch.mean((output_z500 - target_z500)**2)) * 3183.034912109375 # scale
                            
                            t2m = torch.sqrt(torch.mean(val_loss_surface[0,3,:,:])) * 20.369630813598633
                            u10 = torch.sqrt(torch.mean(val_loss_surface[0,1,:,:])) * 5.156293869018555
                            # The weight of surface loss is 0.25    
                            val_loss = (weighted_upper_loss + weighted_surface_loss * 0.25)/ 6.3
                            
                            print("Step {i} validation loss: {loss}, RMSE: t2m: {t2m}, 10u: {u10}, z500:{z500}".format(i=iters, loss=val_loss.item(), t2m=t2m, u10=u10, z500=rmse_z500))
                    
                            
                            if val_count > 0: # only validate one batch
                                break
                
                # save checkpoint
                new_checkpoint_path = checkpoint_path.replace('.tar', '_valloss{:.3f}_step{}.tar'.format(val_loss.item(), iters))
                if rank == 0:
                    save_checkpoint(model.module, None, new_checkpoint_path, iters, i, optimizer, lr_scheduler, train_loader, True, logger)
        
        epoch_loss /= max(1,len(train_loader))
        print(epoch_loss)

        if rank == 0:
            logger.info("Epoch {} : {:.3f}".format(i, epoch_loss))
        loss_list.append(epoch_loss)
        lr_scheduler.step()
        # scaler.update(lr_scheduler)
        #
        # for name, param in model.named_parameters():
        #   writer.add_histogram(name, param.data, i)


        # # Begin to validate
        # if i % cfg.PG.VAL.INTERVAL == 0:
        #     with torch.no_grad():
        #         model.eval()
        #         val_loss = 0.0
        #         for id, val_data in enumerate(val_loader, 0):
        #             input_val, input_surface_val, target_val, target_surface_val, periods_val = val_data
        #             input_val_raw, input_surface_val_raw = input_val, input_surface_val
        #             input_val, input_surface_val, target_val, target_surface_val = input_val.to(
        #                 device), input_surface_val.to(device), target_val.to(device), target_surface_val.to(device)

        #             # Inference
        #             output_val, output_surface_val = model(input_val, input_surface_val,
        #                                                    aux_constants['weather_statistics'],
        #                                                    aux_constants['constant_maps'], aux_constants['const_h'])
        #             # Noralize the gt to make the loss compariable
        #             target_val, target_surface_val = utils_data.normData(target_val, target_surface_val,
        #                                                       aux_constants['weather_statistics_last'])

        #             val_loss_surface = criterion(output_surface_val, target_surface_val)
        #             weighted_val_loss_surface = torch.mean(val_loss_surface * surface_weights)

        #             val_loss_upper = criterion(output_val, target_val)
        #             weighted_val_loss_upper = torch.mean(val_loss_upper * upper_weights)

        #             loss = weighted_val_loss_upper + weighted_val_loss_surface * 0.25

        #             val_loss += loss.item()

        #         val_loss /= len(val_loader)
        #         writer.add_scalars('Loss',
        #                            {'train': epoch_loss,
        #                             'val': val_loss},
        #                            i)
        #         logger.info("Validate at Epoch {} : {:.3f}".format(i, val_loss))
        #         # Visualize the training process
        #         png_path = os.path.join(res_path, "png_training")
        #         utils.mkdirs(png_path)
        #         # """
        #         # Normalize the data back to the original space for visualization
        #         output_val, output_surface_val = utils_data.normBackData(output_val, output_surface_val,
        #                                                       aux_constants['weather_statistics_last'])
        #         target_val, target_surface_val = utils_data.normBackData(target_val, target_surface_val,
        #                                                       aux_constants['weather_statistics_last'])

        #         utils.visuailze(output_val.detach().cpu().squeeze(),
        #                         target_val.detach().cpu().squeeze(),
        #                         input_val_raw.squeeze(),
        #                         var='u',
        #                         z=12,
        #                         step=i,
        #                         path=png_path)
        #         utils.visuailze_surface(output_surface_val.detach().cpu().squeeze(),
        #                                 target_surface_val.detach().cpu().squeeze(),
        #                                 input_surface_val_raw.squeeze(),
        #                                 var='msl',
        #                                 step=i,
        #                                 path=png_path)
        #         # Early stopping
        #         if val_loss < best_loss:
        #             best_loss = val_loss
        #             best_model = copy.deepcopy(model)
        #             # Save the best model
        #             torch.save(best_model, os.path.join(model_save_path, 'best_model.pth'))
        #             logger.info(
        #                 f"current best model is saved at {i} epoch.")
        #             epochs_since_last_improvement = 0
        #         else:
        #             epochs_since_last_improvement += 1
        #             if epochs_since_last_improvement >= 5:
        #                 logger.info(
        #                     f"No improvement in validation loss for {epochs_since_last_improvement} epochs, terminating training.")
        #                 break

        # print("lr",lr_scheduler.get_last_lr()[0])
    return best_model

def validate(model, val_loader, device, rank, logger, checkpoint_path=None):
    
    # Loss function
    criterion_val = nn.MSELoss(reduction='none')


    # Load constants and teleconnection indices
    aux_constants = utils_data.loadAllConstants(
        device=device)  # 'weather_statistics','weather_statistics_last','constant_maps','tele_indices','variable_weights'
    upper_weights, surface_weights = aux_constants['variable_weights'] # use the same weights in training and validation

    iters = 0
    
    step_list = [50, 100, 200, 400, 700, 1000, 1500, 2000]
    
    results = {}
    for step in step_list:
        model, _, _, _, _ = restore_checkpoint(checkpoint_path, None, model, None, None, logger,
                        load_optimizer=False, load_scheduler=False, load_counters=False,
                        load_dataloader=False, checkpoint_mode="standard", log_to_screen=True, step=step)
    
        model = model.to(device)
                    
        # validation
        with torch.inference_mode():
            with torch.no_grad():
                val_count = 0
                for val_data in val_loader:
                    
                    timestamp = val_loader.dataset.data.data.time[val_count].values
                    val_count += 1
                    
                    input, target = val_data
                    input = torch.tensor(input, dtype=torch.float32)
                    target = torch.tensor(target, dtype=torch.float32)
                    input_surface = torch.tensor(input[:,:4,:,:], dtype=torch.float32)
                    input = input[:,4:,:,:].view(1,-1,13,721,1440)
                    # print("input shape 0", input.shape)
                    target_surface = torch.tensor(target[:,:4,:,:], dtype=torch.float32)
                    target = target[:,4:,:,:].view(1,-1,13,721,1440)
                    # print("target shape 0", target.shape)
                    input, target, input_surface, target_surface = input.to(device), \
                        target.to(device), input_surface.to(device), \
                        target_surface.to(device)

                    with torch.autocast(device_type='cuda', dtype=torch.float32):
                    # /with torch.cuda.amp.autocast():
                        model.eval()

                        # Note the input and target need to be normalized (done within the function)
                        # Call the model and get the output
                        output, output_surface = model(input, input_surface, 
                                        aux_constants['constant_maps'],
                                        aux_constants['const_h'])  # (1,5,13,721,1440)

                        
                    # We use the MAE loss to train the model
                    # Different weight can be applied for different fields if needed
                    val_loss_surface = criterion_val(output_surface, target_surface)
                    weighted_surface_loss = torch.mean(val_loss_surface * surface_weights)

                    val_loss_upper = criterion_val(output, target)
                    weighted_upper_loss = torch.mean(val_loss_upper * upper_weights)
                    output_z500 = output[0,0,7,:,:]  # 50, 100, 150, 200, 250, 300, 400, 500, 600, ...
                    target_z500 = target[0,0,7,:,:]
                    rmse_z500 = torch.sqrt(torch.mean((output_z500 - target_z500)**2))
                    
                    t2m = torch.sqrt(torch.mean(val_loss_surface[0,3,:,:]))
                    u10 = torch.sqrt(torch.mean(val_loss_surface[0,1,:,:]))
                    # The weight of surface loss is 0.25    
                    val_loss = weighted_upper_loss + weighted_surface_loss * 0.25
                    
                    print("Step {i} validation RMSE at {timestamp}: t2m: {t2m}, 10u: {u10}, z500:{z500}".format(i=iters, timestamp=timestamp, t2m=t2m, u10=u10, z500=rmse_z500))
                    
                    if val_count > 12: # only validate one batch
                        break
                

def test(test_loader, model, device, res_path):
    # set up empty dics for rmses and anormaly correlation coefficients
    rmse_upper_z, rmse_upper_q, rmse_upper_t, rmse_upper_u, rmse_upper_v = dict(), dict(), dict(), dict(), dict()
    rmse_surface = dict()

    acc_upper_z, acc_upper_q, acc_upper_t, acc_upper_u, acc_upper_v = dict(), dict(), dict(), dict(), dict()
    acc_surface = dict()

    # Load all statistics and constants
    aux_constants = utils_data.loadAllConstants(device=device)

    batch_id = 0
    for id, data in enumerate(test_loader, 0):
        # Store initial input for different models
        print(f"predict on {id}")
        input_test, input_surface_test, target_test, target_surface_test, periods_test = data
        input_test, input_surface_test, target_test, target_surface_test = \
            input_test.to(device), input_surface_test.to(device), target_test.to(device), target_surface_test.to(device)
        model.eval()

        # Inference 
        # forward(self, input, input_surface, statistics, maps,const_h)
        output_test, output_surface_test = model(input_test, input_surface_test,
                                                 aux_constants['weather_statistics'],
                                                 aux_constants['constant_maps'], aux_constants['const_h'])
        # Transfer to the output to the original data range
        output_test, output_surface_test = utils_data.normBackData(output_test, output_surface_test,
                                                        aux_constants['weather_statistics_last'])

        target_time = periods_test[1][batch_id]

        # Visualize
        png_path = os.path.join(res_path, "png")
        utils.mkdirs(png_path)

        utils.visuailze(output_test.detach().cpu().squeeze(),
                                target_test.detach().cpu().squeeze(), 
                                input_test.detach().cpu().squeeze(),
                                var='t',
                                z = 2,
                                step=target_time, 
                                path=png_path)
        #['msl', 'u','v','t2m']
        utils.visuailze_surface(output_surface_test.detach().cpu().squeeze(),
                            target_surface_test.detach().cpu().squeeze(),
                            input_surface_test.detach().cpu().squeeze(),
                            var='u10',
                            step=target_time,
                            path=png_path)
  

        # Compute test scores
        # rmse
        output_test = output_test.squeeze()
        target_test = target_test.squeeze()
        output_surface_test = output_surface_test.squeeze()
        target_surface_test = target_surface_test.squeeze()

        rmse_upper_z[target_time] = score.weighted_rmse_torch_channels(output_test[0],
                                                                       target_test[0]).detach().cpu().numpy()
        rmse_upper_q[target_time] = score.weighted_rmse_torch_channels(output_test[1],
                                                                       target_test[1]).detach().cpu().numpy()
        rmse_upper_t[target_time] = score.weighted_rmse_torch_channels(output_test[2],
                                                                       target_test[2]).detach().cpu().numpy()
        rmse_upper_u[target_time] = score.weighted_rmse_torch_channels(output_test[3],
                                                                       target_test[3]).detach().cpu().numpy()
        rmse_upper_v[target_time] = score.weighted_rmse_torch_channels(output_test[4],
                                                                       target_test[4]).detach().cpu().numpy()

        rmse_surface[target_time] = score.weighted_rmse_torch_channels(output_surface_test,
                                                                       target_surface_test).detach().cpu().numpy()


        # acc
        surface_mean, _, upper_mean, _ = aux_constants['weather_statistics_last']
        output_test_anomaly = output_test - upper_mean.squeeze(0)
        output_surface_test_anomaly = output_surface_test - surface_mean.squeeze(0)
        target_test_anomaly = target_test - upper_mean.squeeze(0)
        target_surface_test_anomaly = target_surface_test - surface_mean.squeeze(0)

        acc_upper_z[target_time] = score.weighted_acc_torch_channels(output_test_anomaly[0],
                                                                     target_test_anomaly[0]).detach().cpu().numpy()
        acc_upper_q[target_time] = score.weighted_acc_torch_channels(output_test_anomaly[1],
                                                                     target_test_anomaly[1]).detach().cpu().numpy()
        acc_upper_t[target_time] = score.weighted_acc_torch_channels(output_test_anomaly[2],
                                                                     target_test_anomaly[2]).detach().cpu().numpy()
        acc_upper_u[target_time] = score.weighted_acc_torch_channels(output_test_anomaly[3],
                                                                     target_test_anomaly[3]).detach().cpu().numpy()
        acc_upper_v[target_time] = score.weighted_acc_torch_channels(output_test_anomaly[4],
                                                                     target_test_anomaly[4]).detach().cpu().numpy()

        acc_surface[target_time] = score.weighted_acc_torch_channels(output_surface_test_anomaly,
                                                                     target_surface_test_anomaly).detach().cpu().numpy()
    # Save rmses to csv
    csv_path = os.path.join(res_path, "csv")
    utils.mkdirs(csv_path)
    utils.save_errorScores(csv_path, rmse_upper_z, rmse_upper_q, rmse_upper_t, rmse_upper_u, rmse_upper_v, rmse_surface,
                     "rmse")
    utils.save_errorScores(csv_path, acc_upper_z, acc_upper_q, acc_upper_t, acc_upper_u, acc_upper_v, acc_surface, "acc")


if __name__ == "__main__":
    pass
