import torch
from torch.nn.functional import normalize
import math
from utils.sh_utils import SH2RGB, RGB2SH

def shade(viewpoint_camera, pc, light_pos = None, light_color = None, lighting_optimization = None):

    torch.manual_seed(7)

    view_pos = viewpoint_camera.camera_center # torch.from_numpy(viewpoint_camera.camera_center).to(viewpoint_camera.data_device)

    radiance_multiplier = 200
    nof_lights = 200

    upper_coordinates = torch.quantile(pc.get_xyz, 0.95, dim=0)
    lower_coordinates = torch.quantile(pc.get_xyz, 0.05, dim=0)


    if light_pos is None:
        light_pos = torch.rand(nof_lights, 3, dtype=torch.float32)*torch.tensor([upper_coordinates[0].item()-lower_coordinates[0].item(), upper_coordinates[1].item()-lower_coordinates[1].item(), upper_coordinates[2].item()-lower_coordinates[2].item()])+torch.tensor([lower_coordinates[0].item(), lower_coordinates[1].item(), lower_coordinates[2].item()])
        light_pos = light_pos.to(viewpoint_camera.data_device)
        #light_pos = torch.rand(nof_lights, 3, dtype=torch.float32)*torch.tensor([20.0, 8.0, 18.0])+torch.tensor([-12.0, -2.0, -1.0])
        #light_pos = light_pos.to(viewpoint_camera.data_device)
        #light_pos = torch.tensor([[-7.0, 2.4, 5.5]], dtype=torch.float32).to(viewpoint_camera.data_device) # left
        #light_pos = torch.tensor([[-2.8, 5.0, 8.1]], dtype=torch.float32).to(viewpoint_camera.data_device) #down
        #light_pos = torch.tensor([[-2.8, 0.5, 8.1]], dtype=torch.float32).to(viewpoint_camera.data_device) #up
        #light_pos = torch.tensor([[0, 1.5, 0.1]], dtype=torch.float32).to(viewpoint_camera.data_device)
    #light_color = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32).to(viewpoint_camera.data_device)
    if light_color is None:
        light_color = torch.rand(nof_lights, 3, dtype=torch.float32).to(viewpoint_camera.data_device)
        #light_color = torch.ones(nof_lights, 3, dtype=torch.float32).to(viewpoint_camera.data_device)
        #light_color -= torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32).to(viewpoint_camera.data_device)
        #light_color = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32).to(viewpoint_camera.data_device)

    #if lighting_optimization is not None:
    #    light_color = light_color.requires_grad_(True)

    frag_pos = pc.get_xyz.unsqueeze(1)

    base_color = torch.pow(torch.clip(SH2RGB(pc.get_bc), 0, 1), 2.2) 

    #metallic = torch.clip(SH2RGB(pc.get_mro[:, :, 2].unsqueeze(1)), 0, 1)
    #roughness = torch.clip(SH2RGB(pc.get_mro[:, :, 1].unsqueeze(1)), 0, 1)

    metallic = torch.clip(SH2RGB(pc.get_mro[:, :, 1].unsqueeze(1)), 0, 1)
    roughness = torch.clip(SH2RGB(pc.get_mro[:, :, 0].unsqueeze(1)), 0, 1)
    #ao = torch.clip(SH2RGB(pc.get_mro[:, :, 0].unsqueeze(1)), 0, 1)

    N = torch.clip(SH2RGB(pc.get_normal), 0, 1)
    N = N*2-1
    N = normalize(N, dim=2)
    V = normalize(view_pos-frag_pos, dim=2)

    F0 = torch.tensor(0.04, dtype=torch.float32).to(viewpoint_camera.data_device)
    if lighting_optimization is not None:
        F0 = F0.requires_grad_(True)
    F0 = F0*(1-metallic)+base_color*metallic

    Lo = torch.zeros_like(base_color)
    if lighting_optimization is not None:
        Lo = Lo.requires_grad_(True)

    for i in range(light_pos.shape[0]):
        L = normalize(light_pos[i] - frag_pos, dim=2)
        H = normalize(V + L, dim=2)
        distance = torch.linalg.norm(light_pos[i] - frag_pos, dim=2).unsqueeze(1)
        attenuation = 1.0 / (distance * distance)
        radiance = torch.clip(light_color[i], 0, 1) * attenuation * radiance_multiplier #* (distance<2).to(torch.float32) #* (distance<1.5).to(torch.float32)

        NDF = DistributionGGX(N, H, roughness)
        G   = GeometrySmith(N, V, L, roughness)
        F = fresnelSchlick(torch.clip(torch.sum(H*V, axis=2).unsqueeze(2), min=0.0), F0)

        numerator    = NDF * G * F
        denominator = 4.0 * torch.clip(torch.sum(N*V, axis=2).unsqueeze(2), min=0.0) * torch.clip(torch.sum(N*L, axis=2).unsqueeze(2), min=0.0) + 0.0001
        specular = numerator / denominator
        
        kS = F

        kD = 1 - kS

        kD *= 1.0 - metallic  

        NdotL = torch.clip(torch.sum(N*L, axis=2).unsqueeze(2), min=0.0)     

        Lo = Lo + (kD * base_color / math.pi + specular) * radiance * NdotL

        #Lo = Lo + (specular) * radiance * NdotL
    
    ambient = 0.1 * base_color  #* ao
    
    color = ambient + Lo

    color = color / (color + 1)

    frag_color = torch.pow(color+0.0001, 1.0/2.2) # torch.pow(color, 1.0/2.2)

    pc.shading = RGB2SH(frag_color)



def DistributionGGX(N, H, roughness):

    a = roughness*roughness
    a2 = a*a
    NdotH = torch.clip(torch.sum(N*H, axis=2).unsqueeze(2), min=0.0)
    NdotH2 = NdotH*NdotH

    nom   = a2
    denom = (NdotH2 * (a2 - 1.0) + 1.0)
    denom = math.pi * denom * denom + 0.0001

    return nom / denom


def GeometrySchlickGGX(NdotV, roughness):

    r = (roughness + 1.0)
    k = (r*r) / 8.0

    nom   = NdotV
    denom = NdotV * (1.0 - k) + k

    return nom / denom


def GeometrySmith(N, V, L, roughness):

    NdotV = torch.clip(torch.sum(N*V, axis=2).unsqueeze(2), min=0.0)
    NdotL = torch.clip(torch.sum(N*L, axis=2).unsqueeze(2), min=0.0)
    ggx2 = GeometrySchlickGGX(NdotV, roughness)
    ggx1 = GeometrySchlickGGX(NdotL, roughness)

    return ggx1 * ggx2



def fresnelSchlick(cos_theta, F0):

    return F0 + (1.0 - F0) * torch.pow(torch.clip(1.0 - cos_theta, min=0.0, max=1.0), 5.0)
