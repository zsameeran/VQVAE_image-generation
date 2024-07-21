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

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, "0" to  "7" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"

writer = SummaryWriter(logdir='/tensor_board')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    transforms.Resize((128, 128)), 
    transforms.ToTensor(),
          
])

custom_dataset.transform = transform


batch_size = 4
data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

for batch in data_loader:
   
    selected_images = batch[:10]  
   
    for i, image in enumerate(selected_images):
       
        image_np = image.permute(1, 2, 0).numpy()  
        plt.imshow(image_np)
        plt.axis('off')
        
        save_folder = "Images_to_save"
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

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
          
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings , encoding_indices 


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
        print(z.shape)
        loss, quantized, perplexity, encodings , indices= self._vq_vae(z)
        #return quantized
        x_recon = self._decoder(quantized)

        return loss, x_recon, perplexity , encodings , indices , quantized
    

model = Model().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0003, amsgrad=False)
trained_model = torch.load("/bs_4_fd_7am.pt")
model_new = Model().to(device)
model_new.load_state_dict(trained_model)
model_new
loaded_encoder = model_new._encoder.to(device)
loaded_decoder=model_new._decoder.to(device)
loaded_vqvae=model_new._vq_vae.to(device)
  
      
class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)

fm = 256
net = nn.Sequential(
    MaskedConv2d('A', 512,  fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
    MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
    MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
    MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
    MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
    MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
    MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
    MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
    nn.Conv2d(fm, 512, 1),
    nn.ReLU(True)
    )

net.cuda()

find_loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    list(net.parameters()),
    lr=0.003)
num_epochs = 1
    
            
for epoch in range (num_epochs):
    net.train()
    for i, img in enumerate(data_loader):
        if img.shape[0] < 4:
            break
        img = img.to(device)
        optimizer.zero_grad()
        
        image = loaded_encoder(img)
                        
        image=model_new._pre_vq_conv(image)
        _ ,quantization, _, _,indices = loaded_vqvae(image)
          
        tatpurte_encodings = torch.zeros(indices.shape[0], loaded_vqvae._num_embeddings, device=device )
        tatpurte_encodings.scatter_(1,indices, 1)
        
        indices_reshaped = tatpurte_encodings.view(4,32 ,32 , 512).permute(0 , 3 , 1 , 2)
  
        output_rand = net(indices_reshaped.to(device))
        output_rand = output_rand.permute(0, 2 , 3 , 1)
        output_rand = output_rand.view(-1,512)
        
        pix_out_img = torch.matmul(output_rand, loaded_vqvae._embedding.weight).view(4,32,32,64)
        pix_out_img= pix_out_img.permute(0,3,1,2)
        final_image = loaded_decoder(pix_out_img)

        p_cnn_loss = find_loss(final_image,img)
        
        writer.add_scalar('Loss/train', p_cnn_loss.item())
        
        p_cnn_loss.backward()
        optimizer.step()
        if i%500 == 0:
            print(f"epoch: {epoch}  i : {i} p_cnn_loss:  {p_cnn_loss}")       
  
  
torch.save(net.state_dict(), '/pcnn_logits_11:45_pm.pt')  
