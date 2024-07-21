"""
Vector Quantized- VAE
Sameeran Zingre   
"""
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
import pickle
from collections import Counter

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, "0" to  "7" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_list = [file for file in os.listdir(data_dir) if file.endswith('.jpg')]
        self.pixel_values = [] 

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.image_list[idx])
        image = Image.open(img_name)
        
        if self.transform:
            image = self.transform(image)

        self.pixel_values.append(image.numpy().flatten())
            
        return image
    
        


data_dir = "/Train_data"
custom_dataset = CustomDataset(data_dir)

transform = transforms.Compose([
    transforms.Resize((128,128)), 
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.RandomRotation(degrees=10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomPerspective(distortion_scale=0.1, p=0.1)          
])

custom_dataset.transform = transform

writer = SummaryWriter(logdir='/tensor_board')


batch_size = 2
data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

for batch in data_loader:
   
    selected_images = batch[:10]  
   
    for i, image in enumerate(selected_images):
       
        image_np = image.permute(1, 2, 0).numpy()  
        plt.imshow(image_np)
        plt.axis('off')
        
        save_folder = "/Images_to_save"
        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(os.path.join(save_folder, f"image_{i}.jpg"))
  
        plt.show()
        plt.close()
        
    break

all_pixel_values = np.concatenate(custom_dataset.pixel_values)

data_variance = np.var(all_pixel_values / 255.0)

class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )
    
    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(3,
                                 out_channels=64,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=64,
                                 out_channels=128,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=128,
                                 out_channels=128,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=128,
                                             num_hiddens=128,
                                             num_residual_layers=2,
                                             num_residual_hiddens=32)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)
        
        x = self._conv_2(x)
        x = F.relu(x)
        
        x = self._conv_3(x)
        return self._residual_stack(x)
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self._conv_1 = nn.Conv2d(in_channels=64,
                                 out_channels=128,
                                 kernel_size=3, 
                                 stride=1, padding=1)
        
        self._residual_stack = ResidualStack(in_channels=128,
                                             num_hiddens=128,
                                             num_residual_layers=2,
                                             num_residual_hiddens=32)
        
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=128, 
                                                out_channels=64,
                                                kernel_size=4, 
                                                stride=2, padding=1)
        
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=64, 
                                                out_channels=3,
                                                kernel_size=4, 
                                                stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        
        x = self._residual_stack(x)
        
        x = self._conv_trans_1(x)
        x = F.relu(x)
        
        return self._conv_trans_2(x)
class VectorQuantizerEMA(nn.Module):
    def __init__(self):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = 64
        self._num_embeddings = 512
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = 0.25
        
        self.register_buffer('_ema_cluster_size', torch.zeros(self._num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(self._num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = 0.99
        self._epsilon = 1e-5

    def forward(self, inputs):
      
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
    
        flat_input = inputs.view(-1, self._embedding_dim)
        
    
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
 
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
    
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            

            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
    
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        

        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encoding_indices


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self._encoder = Encoder()
        self._pre_vq_conv = nn.Conv2d(in_channels=128, 
                                      out_channels=64,
                                      kernel_size=1, 
                                      stride=1)

        self._vq_vae = VectorQuantizerEMA()
 
        self._decoder = Decoder()

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, indices = self._vq_vae(z)
        #return quantized
        x_recon = self._decoder(quantized)

        return loss, x_recon, perplexity,indices
    


model = Model().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0003, amsgrad=False)
model.train()
epochs = 50
j = 0
train_res_recon_error = []
train_res_perplexity = []
latent_encoding = []
latent_e_indices = [] 
for i in range(epochs):
    for img in data_loader:
        data = img.to(device)
        optimizer.zero_grad()

        vq_loss, data_recon, perplexity,indices = model(data)
        recon_error = F.mse_loss(data_recon, data) / data_variance
        loss = recon_error + vq_loss
        loss.backward()

        optimizer.step()
        
        latent_e_indices.append(indices)
        
        train_res_recon_error.append(recon_error.item())
        train_res_perplexity.append(perplexity.item())
        
    
    print('%d epochs' % (i+1))
    print('recon_error: %.3f' % recon_error.item())
    print('perplexity: %.3f' % perplexity.item())
    image_np = data_recon[0].detach().cpu().squeeze().permute(1, 2, 0).numpy()  
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

    plt.imshow(image_np)
    plt.axis('off')
    
    writer.add_scalar('Loss/train', loss.item(), j)
    writer.add_scalar('Reconstruction Error/train', recon_error.item(), j)
    writer.add_scalar('Perplexity/train', perplexity.item(), j)
    j += 1  
    save_folder = "/T_image"
    os.makedirs(save_folder, exist_ok=True)
    plt.savefig(os.path.join(save_folder, f"datta_Image_{i}.jpg"))
    plt.show()
    plt.close()


torch.save(model.state_dict(), '/bs_4_fd_7am.pt')

with open('latent_encoding.pkl', 'wb') as f:
    pickle.dump(latent_encoding, f)

with open('latent_e_indices.pkl', 'wb') as f:
    pickle.dump(latent_e_indices, f)

flat_indices = [idx for sublist in latent_e_indices for idx in sublist]


index_counts = Counter(flat_indices)


indices = list(index_counts.keys())
counts = list(index_counts.values())


plt.figure(figsize=(10, 6))
plt.scatter(indices, counts, marker='o', color='b')
plt.xlabel('Index')
plt.ylabel('Frequency')
plt.title('Distribution of Encoding Indices (Codebook)')
plt.grid(True)


save_folder = "/T_image"
os.makedirs(save_folder, exist_ok=True)
dot_plot_path = os.path.join(save_folder, "encoding_indices_dot_plot.jpg")
plt.savefig(dot_plot_path)


writer.add_image("Encoding Indices Dot Plot", np.array(plt.gcf().canvas.renderer.buffer_rgba()), global_step=0)

plt.close()
writer.close()
