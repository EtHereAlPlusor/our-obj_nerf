import torch
import torch.nn as nn
from lib.config.config import cfg

# Positional encoding (section 5.1)
class Embedder: # 功能就是进行Positional encoding 将输入的d维向量按γ(p)升维成高维向量，并限制在[-1,1]
    def __init__(self, **kwargs):
        self.kwargs = kwargs # 命令行参数 以字典的形式存放到kwargs中
        self.create_embedding_fn()
        
    def create_embedding_fn(self): # 相当于构造函数
        embed_fns = []
        d = self.kwargs['input_dims'] # d为输入的维度 x,d应该都是3维
        out_dim = 0 # 输出的维度 不include_input的话 输出应该是3*(2*N_freqs) = 6 * L
        if self.kwargs['include_input']: # 按论文应该不包括 这使得输入会包含一项原始输入x 看下面程序又包括了，奇怪
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2'] # 对应论文的log(2^(L-1)*pi) 看下面的程序好像就是 L-1
        N_freqs = self.kwargs['num_freqs'] # 对应论文的L
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs) # 按论文应该按这个方式取样频带
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands: # freq 从 2^0*pi变化到 2^(L-1)*pi
            for p_fn in self.kwargs['periodic_fns']: # 指定了周期变化的函数形式 按照论文中应为 [sin(), cos()]
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq)) # 依次添加相应频率的三角函数到embed_fns中
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs): # 将输入依次放入embed_fns的函数中得到相应的输出，并拼接。输出维度应为out_dim=d*2*L
        with torch.no_grad():
            return torch.cat([fn(inputs) for fn in self.embed_fns], -1) 


def get_embedder(multires, i=0): # 返回一个进行PE的函数embed()，以及其相应的输出维度
    if i == -1:
        return nn.Identity(), 3 # nn.Identity()就是原样返回输入的一个函数
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1, # L-1
                'num_freqs' : multires,       # L
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim