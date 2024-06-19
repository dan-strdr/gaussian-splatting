import torch
from torch.nn.functional import normalize
import math
from utils.sh_utils import SH2RGB, RGB2SH

def shade(viewpoint_camera, pc):

    #view_pos = torch.from_numpy(viewpoint_camera.camera_center).to(viewpoint_camera.data_device)
    view_pos = torch.from_numpy(viewpoint_camera.camera_position).to(viewpoint_camera.data_device)

    print('viewpoint_camera.T', viewpoint_camera.T)
    print('viewpoint_camera.camera_center', viewpoint_camera.camera_center)
    print('viewpoint_camera.camera_position', viewpoint_camera.camera_position)

    light_pos = torch.tensor([0, 1.5, 0.1], dtype=torch.float32).to(viewpoint_camera.data_device)
    light_color = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).to(viewpoint_camera.data_device)

    frag_pos = pc.get_xyz.unsqueeze(1)
    
    #print('pc.get_bc.shape', pc.get_bc.shape, pc.get_bc.dtype, torch.isnan(pc.get_bc).sum())
    #print('pc.get_bc', pc.get_bc)
    #print('SH2RGB(pc.get_bc)', SH2RGB(pc.get_bc).max(axis=0), SH2RGB(pc.get_bc).min(axis=0), torch.topk(SH2RGB(pc.get_bc), dim=0, k=10))

    base_color = torch.pow(torch.clip(SH2RGB(pc.get_bc), 0, 1), 2.2) # torch.pow(SH2RGB(pc.get_bc), 2.2)

    #print('SH2RGB_shading(base_color)', SH2RGB_shading(base_color), SH2RGB_shading(base_color).shape)
    #print('base_color.shape', base_color.shape, base_color.dtype, torch.isnan(base_color).sum())
    metallic = torch.clip(SH2RGB(pc.get_mro[:, :, 2].unsqueeze(1)), 0, 1)
    roughness = torch.clip(SH2RGB(pc.get_mro[:, :, 1].unsqueeze(1)), 0, 1)
    #ao = pc.get_mro[:, :, 0].unsqueeze(1)

    N = torch.clip(SH2RGB(pc.get_normal), 0, 1)
    N = N*2-1
    V = normalize(view_pos-frag_pos, dim=2)

    print('V:', V)

    print('V.shape', V.shape, V.dtype, torch.isnan(V).sum())



    F0 = torch.tensor(0.04, dtype=torch.float32).to(viewpoint_camera.data_device)
    F0 = F0*(1-metallic)+base_color*metallic

    Lo = torch.zeros_like(base_color)


    L = normalize(light_pos - frag_pos, dim=2)
    H = normalize(V + L, dim=2)
    distance = torch.linalg.norm(light_pos - frag_pos, dim=2).unsqueeze(1)
    attenuation = 1.0 / (distance * distance)
    radiance = light_color * attenuation * 100

    print('radiance.shape', radiance.shape, radiance.dtype, torch.isnan(radiance).sum())
    print('base_color.shape', base_color.shape, base_color.dtype, torch.isnan(base_color).sum())
    print('metallic.shape', metallic.shape, metallic.dtype, torch.isnan(metallic).sum())
    print('N.shape', N.shape, N.dtype, torch.isnan(N).sum())
    print('F0.shape', F0.shape, F0.dtype, torch.isnan(F0).sum())
    print('Lo.shape', Lo.shape, Lo.dtype, torch.isnan(Lo).sum())
    print('L.shape', L.shape, L.dtype, torch.isnan(L).sum())
    print('H.shape', H.shape, H.dtype, torch.isnan(H).sum())
    print('distance.shape', distance.shape, distance.dtype, torch.isnan(distance).sum())
    print('attenuation.shape', attenuation.shape, attenuation.dtype, torch.isnan(attenuation).sum())

    NDF = DistributionGGX(N, H, roughness)
    G   = GeometrySmith(N, V, L, roughness)
    F = fresnelSchlick(torch.clip(torch.sum(H*V, axis=2).unsqueeze(2), min=0.0), F0)

    print('NDF.shape', NDF.shape, NDF.dtype, torch.isnan(NDF).sum())
    print('G.shape', G.shape, G.dtype, torch.isnan(G).sum())
    print('F.shape', F.shape, F.dtype, torch.isnan(F).sum())

    numerator    = NDF * G * F
    denominator = 4.0 * torch.clip(torch.sum(N*V, axis=2).unsqueeze(2), min=0.0) * torch.clip(torch.sum(N*L, axis=2).unsqueeze(2), min=0.0) + 0.0001
    specular = numerator / denominator

    print('specular.shape', specular.shape, specular.dtype, torch.isnan(specular).sum())
    
    kS = F

    kD = 1 - kS

    kD *= 1.0 - metallic  

    NdotL = torch.clip(torch.sum(N*L, axis=2).unsqueeze(2), min=0.0)     

    print('kD.shape', kD.shape, kD.dtype, torch.isnan(kD).sum())
    print('kS.shape', kS.shape, kS.dtype, torch.isnan(kS).sum())
    print('NdotL.shape', NdotL.shape, NdotL.dtype, torch.isnan(NdotL).sum())

    Lo += (kD * base_color / math.pi + specular) * radiance * NdotL
    
    ambient = 0.3 * base_color  # * ao
    
    color = ambient + Lo

    color = color / (color + 1)

    frag_color = torch.pow(color, 1.0/2.2) # torch.pow(color, 1.0/2.2)

    pc.shading = RGB2SH(frag_color)



def DistributionGGX(N, H, roughness):

    a = roughness*roughness
    a2 = a*a
    NdotH = torch.clip(torch.sum(N*H, axis=2).unsqueeze(2), min=0.0)
    NdotH2 = NdotH*NdotH

    nom   = a2
    denom = (NdotH2 * (a2 - 1.0) + 1.0)
    print('a.shape', a.shape, a.dtype, torch.isnan(a).sum())
    print('a2.shape', a2.shape, a2.dtype, torch.isnan(a2).sum())
    print('NdotH.shape', NdotH.shape, NdotH.dtype, torch.isnan(NdotH).sum())
    print('NdotH2.shape', NdotH2.shape, NdotH2.dtype, torch.isnan(NdotH2).sum())
    print('denom.shape', denom.shape, denom.dtype, torch.isnan(denom).sum())
    denom = math.pi * denom * denom + 0.0001

    return nom / denom


def GeometrySchlickGGX(NdotV, roughness):

    r = (roughness + 1.0)
    k = (r*r) / 8.0

    nom   = NdotV
    denom = NdotV * (1.0 - k) + k

    print('nom.shape', nom.shape, nom.dtype, torch.isnan(nom).sum())
    print('denom.shape', denom.shape, denom.dtype, torch.isnan(denom).sum())

    return nom / denom


def GeometrySmith(N, V, L, roughness):

    NdotV = torch.clip(torch.sum(N*V, axis=2).unsqueeze(2), min=0.0)
    NdotL = torch.clip(torch.sum(N*L, axis=2).unsqueeze(2), min=0.0)
    ggx2 = GeometrySchlickGGX(NdotV, roughness)
    ggx1 = GeometrySchlickGGX(NdotL, roughness)

    print('ggx2.shape', ggx2.shape, ggx2.dtype, torch.isnan(ggx2).sum())
    print('ggx1.shape', ggx1.shape, ggx1.dtype, torch.isnan(ggx1).sum())

    return ggx1 * ggx2



def fresnelSchlick(cos_theta, F0):

    return F0 + (1.0 - F0) * torch.pow(torch.clip(1.0 - cos_theta, min=0.0, max=1.0), 5.0)
