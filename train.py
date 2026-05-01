import argparse   #Libary for parsing command line arguments
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from pathlib import Path #Libary for working with file paths 
from utils.utils import *
from utils.models import *
from tqdm import tqdm  # tqdm is a Python library that provides a fast, extensible progress bar for loops and other iterable objects. It allows you to visualize the progress of your code execution, making it easier to track the status of long-running tasks. In this code, tqdm is used to create a progress bar for the training loop, which helps to monitor the training process and provides real-time feedback on the loss values.
from torchvision.utils import save_image 


def parse_arguments():
    parser = argparse.ArgumentParser() #object for parsing command line arguments
    
    # Arguments for specifying the locations of the content and style datasets, as well as the pre-trained VGG model
    parser.add_argument('--content_dir', type=str, default="content_data",
                        help='Location of content dataset')
    parser.add_argument('--style_dir', type=str, default="style_data",
                        help='Location of style dataset')
    parser.add_argument('--vgg', type=str, default="vgg_normalised.pth", 
                        help='Location of pre-trained VGG') #pre-trained VGG model is used as encoder for feature extraction
    parser.add_argument('--experiment', type=str, default='experiment1',
                        help='Name of experiment')
    
    # The following arguments specify the sizes for resizing and cropping the content and style images, as well as the final size of the output image. The --crop argument is a boolean flag that indicates whether to apply random cropping to the images during preprocessing.
    parser.add_argument('--final_size', type=int, default=256,
                        help='Size of final image')
    parser.add_argument('--content_size', type=int, default=512,
                        help='Size of content image')
    parser.add_argument('--style_size', type=int, default=512,
                        help='Size of style image')
    parser.add_argument('--crop', action='store_true', default=True,
                        help='Crop image')
    
    # The following arguments specify the batch size, learning rate, learning rate decay, number of epochs, content weight, style weight, log interval, save interval, and options for resuming training from checkpoints. These hyperparameters control the training process and can be adjusted to achieve better results.
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--lr_decay', type=float, default=5e-5,
                        help='Learning rate decay')
    
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs')
    
    parser.add_argument('--content_weight', type=float, default=1.0,
                        help='Content weight')
    parser.add_argument('--style_weight', type=float, default=5,
                        help='Style weight')
    
    parser.add_argument('--log_interval', type=int, default=1,
                        help='Log interval')
    
    parser.add_argument('--save_interval', type=int, default=2,
                        help='Save interval')
    
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume training')
    
    parser.add_argument('--decoder_path', type=str, default=None,
                        help='Path to decoder checkpoint')
    
    parser.add_argument('--optimizer_path', type=str, default=None,
                        help='Path to optimizer checkpoint')
    

    return parser.parse_args()


def main():
    # We can access the command line arguments using the args object, e.g., args.content_dir, args.style_dir, etc.
    args = parse_arguments() 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path('experiment') / args.experiment
    save_dir.mkdir(exist_ok=True, parents=True)

    #Save argument values in a text file for future reference. This is useful for keeping track of the hyperparameters and settings used in each experiment.
    with open(save_dir / 'args.txt', 'w') as args_file:
        for key, value in vars(args).items(): #vars() function returns the __dict__ attribute of the given object, which is a dictionary containing all the attributes of the object. items() method returns a view object that displays a list of a dictionary's key-value tuple pairs.
            args_file.write(f'{key}: {value}\n')
    
    # The get_transform function is called to create transformation pipelines for both content and style images. These transformations include resizing, cropping, and converting the images to tensors. The transformations are applied to the images when they are loaded from the dataset.
    content_transform = get_transform(args.content_size, args.crop, args.final_size)
    style_transform = get_transform(args.style_size, args.crop, args.final_size)
    
    # The ImageFolderDataset class is used to create datasets for both content and style images. The datasets are initialized with the respective directories and transformations. DataLoader objects are then created for both datasets, which allow for efficient batching and shuffling of the data during training.
    content_dataset = ImageFolderDataset(args.content_dir, content_transform)
    style_dataset = ImageFolderDataset(args.style_dir, style_transform)
    
    # The DataLoader class from PyTorch is used to create data loaders for both the content and style datasets. The data loaders are responsible for loading the data in batches, shuffling the data, and optimizing memory usage during training. The batch size, shuffle option, pin_memory option, and drop_last option are specified when creating the data loaders.
    content_dataloader = DataLoader(content_dataset,
                                    batch_size=args.batch_size,
                                    shuffle = True,
                                    pin_memory=True,
                                    drop_last=True) #pin_memory=True allows the data loader to copy tensors into CUDA pinned memory before returning them, which can improve performance when using a GPU. drop_last=True ensures that the last batch is dropped if it is smaller than the specified batch size, which can help maintain consistent batch sizes during training.
    style_dataloader = DataLoader(style_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=True)
    
    # The number of batches in both the content and style datasets is printed to the console. This information can be useful for understanding the size of the datasets and how many iterations will be required during training.
    print('Number of batches in content dataset: ', len(content_dataloader))
    print('Number of batches in style dataset: ', len(style_dataloader))
    
    # Initialize the VGG encoder and decoder models
    encoder = VGGEncoder(args.vgg).to(device)
    decoder = Decoder().to(device)

    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)

    # The LambdaLR scheduler is used to adjust the learning rate during training. The learning rate is decayed according to the formula: lr = initial_lr / (1 + lr_decay * epoch), where initial_lr is the learning rate specified in the command line arguments, lr_decay is the learning rate decay factor, and epoch is the current epoch number. This allows for a gradual decrease in the learning rate as training progresses, which can help improve convergence and prevent overfitting.
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda = lambda epoch:  1.0 / (1.0 + args.lr_decay * epoch)
    )

    if args.resume:
        decoder.load_state_dict(torch.load(args.decoder_path))
        optimizer.load_state_dict(torch.load(args.optimizer_path))

    print('Training...')

    mse_loss = torch.nn.MSELoss()

    encoder.eval()

    running_loss = None #running_loss is initialized to None at the beginning of the training loop. This variable will be used to accumulate the total loss over each epoch, allowing us to calculate the average loss at the end of the epoch. By initializing it to None, we can check if it has been assigned a value during the first iteration of the loop and handle it accordingly when calculating the average loss.
    running_closs = None  #running_closs is initialized to None at the beginning of the training loop. This variable will be used to accumulate the content loss over each epoch, allowing us to calculate the average content loss at the end of the epoch. By initializing it to None, we can check if it has been assigned a value during the first iteration of the loop and handle it accordingly when calculating the average content loss.
    running_sloss = None   #running_sloss is initialized to None at the beginning of the training loop. This variable will be used to accumulate the style loss over each epoch, allowing us to calculate the average style loss at the end of the epoch. By initializing it to None, we can check if it has been assigned a value during the first iteration of the loop and handle it accordingly when calculating the average style loss.

    for epoch in range(args.epochs):
        progress_bar = tqdm(zip(content_dataloader, style_dataloader),
                            total=min(len(content_dataloader), len(style_dataloader)))

        running_loss = 0  #running_loss is reset to 0 at the beginning of each epoch. This variable will be used to accumulate the total loss for the current epoch, allowing us to calculate the average loss at the end of the epoch. By resetting it to 0, we ensure that we are only calculating the loss for the current epoch and not carrying over any values from previous epochs.
        running_closs = 0
        running_sloss = 0
        
        # The training loop iterates over the content and style data loaders simultaneously using the zip function. For each batch of content and style images, the following steps are performed:
        for content_batch, style_batch in progress_bar:

            content_batch = content_batch.to(device) # The content_batch and style_batch tensors are moved to the specified device (GPU or CPU) using the to() method. This allows for efficient computation during training, as the data will be processed on the appropriate hardware. By moving the batches to the device, we can take advantage of GPU acceleration if available, which can significantly speed up the training process.
            style_batch = style_batch.to(device)
            
            c_feats = encoder(content_batch) # The content and style batches are passed through the encoder to extract their respective features. The encoder is a pre-trained VGG model that has been modified to output feature maps at different layers. The extracted features will be used to compute the content and style losses during training.
            s_feats = encoder(style_batch)

            t = adaptive_instance_normalization(c_feats[-1], s_feats[-1])

            g = decoder(t)

            g_feats = encoder(g)

            loss_c = mse_loss(g_feats[-1], t) * args.content_weight

            loss_s = 0
            for g_f, s_f in zip(g_feats, s_feats):
                g_mean, g_std = calc_mean_std(g_f)
                s_mean, s_std = calc_mean_std(s_f)
                loss_s += mse_loss(g_mean, s_mean) + mse_loss(g_std, s_std)
            
            loss_s = loss_s * args.style_weight

            loss = loss_c + loss_s

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # The progress bar is updated with the current loss values for the total loss, content loss, and style loss. This allows us to monitor the training process in real-time and see how the losses are evolving as the model learns. By providing this feedback, we can gain insights into the training dynamics and make adjustments if necessary.
            progress_bar.set_description(f'Loss:{loss.item():4f}, Content Loss: {loss_c.item():4f}, Style Loss: {loss_s.item():4f}')

            running_loss += loss.item()
            running_closs += loss_c.item()
            running_sloss += loss_s.item()
        
        scheduler.step() # The scheduler's step() method is called at the end of each epoch to update the learning rate according to the defined schedule. In this case, the learning rate will be decayed based on the formula specified in the LambdaLR scheduler, which allows for a gradual decrease in the learning rate as training progresses. This can help improve convergence and prevent overfitting by allowing the model to make smaller updates to the weights as it gets closer to an optimal solution.

        running_loss /= len(content_dataloader)
        running_closs /= len(content_dataloader)
        running_sloss /= len(content_dataloader)
        
        # The loss values for the total loss, content loss, and style loss are printed to the console at specified intervals defined by the log_interval argument. This allows us to track the training progress and see how the losses are evolving over time. By providing this information, we can gain insights into the training dynamics and make adjustments if necessary.
        if (epoch+1) % args.log_interval == 0:
            tqdm.write(f'Iter {epoch+1}: Loss:{running_loss:4f}, Content Loss: {running_closs:4f}, Style Loss: {running_sloss:4f}')
        
        # The model's state and optimizer's state are saved at specified intervals defined by the save_interval argument. This allows us to checkpoint the training process and resume it later if needed. Additionally, a sample output image is generated by concatenating the content batch, style batch, and generated images (g) and saving it to the specified directory. This provides a visual representation of the model's performance at different stages of training, allowing us to see how well the style transfer is working.
        if (epoch+1) % args.save_interval == 0:
            torch.save(decoder.state_dict(), save_dir / f'decoder_{epoch+1}.pth')
            torch.save(optimizer.state_dict(), save_dir / f'optimizer_{epoch+1}.pth')

            with torch.no_grad():
                output = torch.cat([content_batch, style_batch, g], dim=0)
                save_image(output, save_dir / f'output_{epoch+1}.png', nrow=args.batch_size)




if __name__ == '__main__':
    main()