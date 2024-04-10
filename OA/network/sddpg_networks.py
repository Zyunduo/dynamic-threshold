import torch
import torch.nn as nn
import numpy as np
import random
import copy
import math



NEURON_VTH_1 = torch.full((1,256),0.5)
NEURON_VTH_2 = torch.full((1,256),0.5)
NEURON_VTH_3 = torch.full((1,256),0.5)
NEURON_VTH_4 = torch.full((1,2),0.5)
NEURON_VTH_1_temporal = torch.full((1,256),0.5)
NEURON_VTH_2_temporal = torch.full((1,256),0.5)
NEURON_VTH_3_temporal = torch.full((1,256),0.5)
NEURON_VTH_4_temporal = torch.full((1,2),0.5)
NEURON_VTH_1_energy = torch.full((1,256),0.5)
NEURON_VTH_2_energy = torch.full((1,256),0.5)
NEURON_VTH_3_energy = torch.full((1,256),0.5)
NEURON_VTH_4_energy = torch.full((1,2),0.5)
NEURON_DYNAMIC_TH_RATE = 0.01
NEURON_CDECAY = 1 / 2
NEURON_VDECAY = 3 / 4
SPIKE_PSEUDO_GRAD_WINDOW = 0.5


class PseudoSpikeRect_1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(NEURON_VTH_1).float()
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        spike_pseudo_grad = (abs(input - NEURON_VTH_1) < SPIKE_PSEUDO_GRAD_WINDOW)
        return grad_input * spike_pseudo_grad.float()

class PseudoSpikeRect_2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(NEURON_VTH_2).float()  
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        spike_pseudo_grad = (abs(input - NEURON_VTH_2) < SPIKE_PSEUDO_GRAD_WINDOW)
        return grad_input * spike_pseudo_grad.float()

class PseudoSpikeRect_3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(NEURON_VTH_3).float()  
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        spike_pseudo_grad = (abs(input - NEURON_VTH_3) < SPIKE_PSEUDO_GRAD_WINDOW)
        return grad_input * spike_pseudo_grad.float()

class PseudoSpikeRect_4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(NEURON_VTH_4).float() 
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        spike_pseudo_grad = (abs(input - NEURON_VTH_4) < SPIKE_PSEUDO_GRAD_WINDOW)
        return grad_input * spike_pseudo_grad.float()

class ActorNetSpiking(nn.Module):
    """ Spiking Actor Network """
    def __init__(self, state_num, action_num, device, batch_window=50, hidden1=256, hidden2=256, hidden3=256):
        """

        :param state_num: number of states
        :param action_num: number of actions
        :param device: device used
        :param batch_window: window steps
        :param hidden1: hidden layer 1 dimension
        :param hidden2: hidden layer 2 dimension
        :param hidden3: hidden layer 3 dimension
        """
        super(ActorNetSpiking, self).__init__()
        self.state_num = state_num
        self.action_num = action_num
        self.device = device
        self.batch_window = batch_window
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.pseudo_spike_1 = PseudoSpikeRect_1.apply
        self.pseudo_spike_2 = PseudoSpikeRect_2.apply
        self.pseudo_spike_3 = PseudoSpikeRect_3.apply
        self.pseudo_spike_4 = PseudoSpikeRect_4.apply
        self.fc1 = nn.Linear(self.state_num, self.hidden1, bias=True)
        self.fc2 = nn.Linear(self.hidden1, self.hidden2, bias=True)
        self.fc3 = nn.Linear(self.hidden2, self.hidden3, bias=True)
        self.fc4 = nn.Linear(self.hidden3, self.action_num, bias=True)

    def neuron_model_1(self, syn_func, pre_layer_output, current, volt_, spike, step):
        """
        Neuron Model
        :param syn_func: synaptic function
        :param pre_layer_output: output from pre-synaptic layer
        :param current: current of last step
        :param volt: voltage of last step
        :param spike: spike of last step
        :return: current, volt, spike
        """
        global NEURON_VTH_1
        global NEURON_VTH_1_temporal
        current = current * NEURON_CDECAY + syn_func(pre_layer_output)
        # volt_ = copy.deepcopy(volt)
        volt = volt_ * NEURON_VDECAY * (1. - spike) + current
        # if step != 0:
            # if volt_.shape[0] == 1:
        #NEURON_VTH_1_energy = np.clip(np.exp((volt_.detach().numpy() - volt.detach().numpy())/0.1) * 0.0999 + 0.2, 0.0, 1.0)
        a = - np.exp(- abs(NEURON_VTH_1.mean())).detach().numpy()
        NEURON_VTH_1_energy = np.exp((volt_.detach().numpy() - volt.detach().numpy())/3) - 1
        NEURON_VTH_1 = 0.5 * NEURON_VTH_1_temporal + 0.5 * torch.from_numpy(NEURON_VTH_1_energy)
            # elif volt_.shape[0] == 256:
                # NEURON_VTH_1_energy = torch.zeros(256,1).numpy()
                # for i in range(256):
                # NEURON_VTH_1_energy = np.clip(np.exp((volt_.mean(dim=1).detach().numpy() - volt.mean(dim=1).detach().numpy())/0.3) * 0.01 + 0.05, 0.0, 1.0).reshape(256,1)
                # NEURON_VTH_1 = 0.48 * NEURON_VTH_1_temporal + 0.48 * torch.from_numpy(NEURON_VTH_1_energy) + random.uniform(-0.001, 0.001)
                # print(type(NEURON_VTH_1))
                # print(NEURON_VTH_1)
        spike = self.pseudo_spike_1(volt)
        return current, volt, spike

    def neuron_model_2(self, syn_func, pre_layer_output, current, volt_, spike, step):
        """
        Neuron Model
        :param syn_func: synaptic function
        :param pre_layer_output: output from pre-synaptic layer
        :param current: current of last step
        :param volt: voltage of last step
        :param spike: spike of last step
        :return: current, volt, spike
        """
        global NEURON_VTH_2
        global NEURON_VTH_2_temporal
        current = current * NEURON_CDECAY + syn_func(pre_layer_output)
        # volt_ = copy.deepcopy(volt)
        volt = volt_ * NEURON_VDECAY * (1. - spike) + current
        # if step != 0:
            # if volt_.shape[0] == 1:
        #NEURON_VTH_2_energy = np.clip(np.exp((volt_.detach().numpy() - volt.detach().numpy())/0.1) * 0.0999 + 0.35, 0.2, 1.0)
        a = - np.exp(- abs(NEURON_VTH_2.mean())).detach().numpy()
        NEURON_VTH_2_energy = np.exp((volt_.detach().numpy() - volt.detach().numpy())/3) - 1
        NEURON_VTH_2 = 0.5 * NEURON_VTH_2_temporal + 0.5 * torch.from_numpy(NEURON_VTH_2_energy)
            # elif volt_.shape[0] == 256:
                # NEURON_VTH_2_energy = torch.zeros(256,1).numpy()
                # for i in range(256):
                # NEURON_VTH_2_energy = np.clip(np.exp((volt_.mean(dim=1).detach().numpy() - volt.mean(dim=1).detach().numpy())/0.3) * 0.01 + 0.05, 0.2, 1.0).reshape(256,1)
                # NEURON_VTH_2 = 0.48 * NEURON_VTH_2_temporal + 0.48 * torch.from_numpy(NEURON_VTH_2_energy) + random.uniform(-0.001, 0.001)
        spike = self.pseudo_spike_2(volt)
        return current, volt, spike
    
    def neuron_model_3(self, syn_func, pre_layer_output, current, volt_, spike, step):
        """
        Neuron Model
        :param syn_func: synaptic function
        :param pre_layer_output: output from pre-synaptic layer
        :param current: current of last step
        :param volt: voltage of last step
        :param spike: spike of last step
        :return: current, volt, spike
        """
        global NEURON_VTH_3
        global NEURON_VTH_3_temporal
        current = current * NEURON_CDECAY + syn_func(pre_layer_output)
        # volt_ = copy.deepcopy(volt)
        volt = volt_ * NEURON_VDECAY * (1. - spike) + current
        # if step != 0:
            # if volt_.shape[0] == 1:
        #NEURON_VTH_3_energy = np.clip(np.exp((volt_.detach().numpy() - volt.detach().numpy())/0.1) * 0.0999 + 0.48, 0.3, 1.0)
        a = - np.exp(- abs(NEURON_VTH_3.mean())).detach().numpy()
        NEURON_VTH_3_energy = np.exp((volt_.detach().numpy() - volt.detach().numpy())/3) - 1
        NEURON_VTH_3 = 0.5 * NEURON_VTH_3_temporal + 0.5 * torch.from_numpy(NEURON_VTH_3_energy)
            # elif volt_.shape[0] == 256:
                # NEURON_VTH_3_energy = torch.zeros(256,1).numpy()
                # for i in range(256):
                # NEURON_VTH_3_energy = np.clip(np.exp((volt_.mean(dim=1).detach().numpy() - volt.mean(dim=1).detach().numpy())/0.3) * 0.01 + 0.05, 0.3, 1.0).reshape(256,1)
                # NEURON_VTH_3 = 0.48 * NEURON_VTH_3_temporal + 0.48 * torch.from_numpy(NEURON_VTH_3_energy) + random.uniform(-0.001, 0.001)
        spike = self.pseudo_spike_3(volt)
        return current, volt, spike
    
    def neuron_model_4(self, syn_func, pre_layer_output, current, volt_, spike, step):
        """
        Neuron Model
        :param syn_func: synaptic function
        :param pre_layer_output: output from pre-synaptic layer
        :param current: current of last step
        :param volt: voltage of last step
        :param spike: spike of last step
        :return: current, volt, spike
        """
        global NEURON_VTH_4
        global NEURON_VTH_4_temporal
        current = current * NEURON_CDECAY + syn_func(pre_layer_output)
        # volt_ = copy.deepcopy(volt)
        volt = volt_ * NEURON_VDECAY * (1. - spike) + current
        # if step != 0:
            # if volt_.shape[0] == 1:
        #NEURON_VTH_4_energy = np.clip(np.exp((volt_.detach().numpy() - volt.detach().numpy())/0.1) * 0.0999 + 0.5, 0.4, 1.0)
        a = - np.exp(- abs(NEURON_VTH_4.mean())).detach().numpy()
        NEURON_VTH_4_energy = np.exp((volt_.detach().numpy() - volt.detach().numpy())/3) - 1 
        NEURON_VTH_4 = 0.5 * NEURON_VTH_4_temporal + 0.5 * torch.from_numpy(NEURON_VTH_4_energy)
            # elif volt_.shape[0] == 256:
                # NEURON_VTH_4_energy = torch.zeros(256,1).numpy()
                # for i in range(256):
                # NEURON_VTH_4_energy = np.clip(np.exp((volt_.mean(dim=1).detach().numpy() - volt.mean(dim=1).detach().numpy())/0.3) * 0.01 + 0.05, 0.4, 1.0).reshape(256,1)
                # NEURON_VTH_4 = 0.48 * NEURON_VTH_4_temporal + 0.48 * torch.from_numpy(NEURON_VTH_4_energy) + random.uniform(-0.001, 0.001)
        spike = self.pseudo_spike_4(volt)
        return current, volt, spike

        

    def forward(self, x, batch_size):
        """

        :param x: state batch
        :param batch_size: size of batch
        :return: out
        """
        global NEURON_VTH_1
        global NEURON_VTH_2
        global NEURON_VTH_3
        global NEURON_VTH_4
        global NEURON_VTH_1_temporal
        global NEURON_VTH_2_temporal
        global NEURON_VTH_3_temporal
        global NEURON_VTH_4_temporal
        x, PSP = x
        fc1_u = PSP[0]
        fc1_v = PSP[1]
        fc1_s = PSP[2]
        fc2_u = PSP[3]
        fc2_v = PSP[4]
        fc2_s = PSP[5]
        fc3_u = PSP[6]
        fc3_v = PSP[7]
        fc3_s = PSP[8]
        fc4_u = PSP[9]
        fc4_v = PSP[10]
        fc4_s = PSP[11]
                
        fc4_sumspike = torch.zeros(batch_size, self.action_num, device=self.device)
        for step in range(self.batch_window):
            input_spike = x[:, :, step]
            # if fc1_v.shape[0] == 1 and step != 0:
            #     NEURON_VTH_1 = np.clip(NEURON_VTH_1 + NEURON_DYNAMIC_TH_RATE * (fc1_v - NEURON_VTH_1).sum(), 0.0, 1.0)
            
            fc1_u, fc1_v, fc1_s = self.neuron_model_1(self.fc1, input_spike, fc1_u, fc1_v, fc1_s, step)
            # fc1_v_= copy.deepcopy(fc1_v)
            # if fc1_v.shape[0] == 1:
                #NEURON_VTH_1_temporal = np.clip(0.001 * (fc1_v - (-3.0)) + 0.2 + 0.1 * torch.log(1 + np.exp((fc1_v - (-3.0))/2)), 0.0, 1.0) 
            V_m1 = fc1_v.mean() - 0.2 * (fc1_v.max()-fc1_v.min())
            V_theta1 = NEURON_VTH_1.mean() - 0.2 * (NEURON_VTH_1.max()-NEURON_VTH_1.min())
            NEURON_VTH_1_temporal = 0.01 * (fc1_v - V_m1) + V_theta1 + torch.log(1 + np.exp((fc1_v - V_m1)/4))
                # with open(filename_1_, 'a') as file_object1_:
                #     file_object1_.write(str(fc1_s.sum().numpy())+"\n")
            # if fc1_v.shape[0] == 256:
                # NEURON_VTH_1_temporal = torch.zeros(256,1).detach().numpy()
                # for i in range(256):
                    # fc1_v_ = copy.deepcopy(fc1_v)
                # NEURON_VTH_1_temporal = np.clip(NEURON_VTH_1 + NEURON_DYNAMIC_TH_RATE * (fc1_v.detach() - NEURON_VTH_1).sum(dim=1).reshape(256,1), 0.0, 1.0)
                    
            #     with open(filename_11, 'a') as file_object11:
            #         file_object11.write(str(self.fc1.weight.data.numpy().mean())+"\n")
            fc2_u, fc2_v, fc2_s = self.neuron_model_2(self.fc2, fc1_s, fc2_u, fc2_v, fc2_s, step)
            # if fc1_v.shape[0] == 1:
                #NEURON_VTH_2_temporal = np.clip(0.001 * (fc2_v - (-10.6)) + 0.35 + 0.1 * torch.log(1 + np.exp((fc2_v - (-10.6))/2)), 0.2, 1.0) 
            V_m2 = fc2_v.mean() - 0.2 * (fc2_v.max()-fc2_v.min())
            V_theta2 = NEURON_VTH_2.mean() - 0.2 * (NEURON_VTH_2.max()-NEURON_VTH_2.min())
            NEURON_VTH_2_temporal = 0.01 * (fc2_v - V_m2) + V_theta2 + torch.log(1 + np.exp((fc2_v - V_m2)/4))
                # with open(filename_2_, 'a') as file_object2_:
                #     file_object2_.write(str(fc2_s.sum().numpy())+"\n")
            # if fc1_v.shape[0] == 256:
                # NEURON_VTH_2_temporal = torch.zeros(256,1).detach().numpy()
                # for i in range(256):
                    # fc2_v_ = copy.deepcopy(fc2_v)
                # NEURON_VTH_2_temporal = np.clip(NEURON_VTH_2 + NEURON_DYNAMIC_TH_RATE * (fc2_v.detach() - NEURON_VTH_2).sum(dim=1).reshape(256,1), 0.2, 1.0)
    
            # if fc1_v.shape[0] == 256:
            #     with open(filename_22, 'a') as file_object22:
            #         file_object22.write(str(self.fc2.weight.data.numpy().mean())+"\n")
            fc3_u, fc3_v, fc3_s = self.neuron_model_3(self.fc3, fc2_s, fc3_u, fc3_v, fc3_s, step)
            # if fc1_v.shape[0] == 1:
                #NEURON_VTH_3_temporal = np.clip(0.001 * (fc3_v - (-12.7)) + 0.48 + 0.1 * torch.log(1 + np.exp((fc3_v - (-12.7))/2)), 0.3, 1.0) 
            V_m3 = fc3_v.mean() - 0.2 * (fc3_v.max()-fc3_v.min())
            V_theta3 = NEURON_VTH_3.mean() - 0.2 * (NEURON_VTH_3.max()-NEURON_VTH_3.min())
            NEURON_VTH_3_temporal = 0.01 * (fc3_v - V_m3) + V_theta3 + torch.log(1 + np.exp((fc3_v - V_m3)/4))
                # with open(filename_3_, 'a') as file_object3_:
                #     file_object3_.write(str(fc3_s.sum().numpy())+"\n")
            # if fc1_v.shape[0] == 256:
                # NEURON_VTH_3_temporal = torch.zeros(256,1).numpy()
                # for i in range(256):
                    # fc3_v_ = copy.deepcopy(fc3_v)
                # NEURON_VTH_3_temporal = np.clip(NEURON_VTH_3 + NEURON_DYNAMIC_TH_RATE * (fc3_v.detach() - NEURON_VTH_3).sum(dim=1).reshape(256,1), 0.3, 1.0)
                
            # if fc1_v.shape[0] == 256:
            #     with open(filename_33, 'a') as file_object33:
            #         file_object33.write(str(self.fc3.weight.data.numpy().mean())+"\n")
            fc4_u, fc4_v, fc4_s = self.neuron_model_4(self.fc4, fc3_s, fc4_u, fc4_v, fc4_s, step)
            # if fc1_v.shape[0] == 1:
                #NEURON_VTH_4_temporal = np.clip(0.001 * (fc4_v - (-8.0)) + 0.5 + 0.1 * torch.log(1 + np.exp((fc4_v - (-8.0))/2)), 0.4, 1.0)
            V_m4 = fc4_v.mean() - 0.2 * (fc4_v.max()-fc4_v.min())
            V_theta4 = NEURON_VTH_4.mean() - 0.2 * (NEURON_VTH_4.max()-NEURON_VTH_4.min()) 
            NEURON_VTH_4_temporal = 0.01 * (fc4_v - V_m4) + V_theta4 + torch.log(1 + np.exp((fc4_v - V_m4)/4)) 
                # with open(filename_4_, 'a') as file_object4_:
                #     file_object4_.write(str(fc4_s.sum().numpy())+"\n")
            # if fc1_v.shape[0] == 256:
                # NEURON_VTH_4_temporal = torch.zeros(256,1).numpy()
                # for i in range(256):
                    # fc4_v_ = copy.deepcopy(fc4_v)
                # NEURON_VTH_4_temporal = np.clip(NEURON_VTH_4 + NEURON_DYNAMIC_TH_RATE * (fc4_v.detach() - NEURON_VTH_4).sum(dim=1).reshape(256,1), 0.4, 1.0)
                
            # if fc1_v.shape[0] == 256:
            #     with open(filename_44, 'a') as file_object44:
            #         file_object44.write(str(self.fc4.weight.data.numpy().mean())+"\n")
            fc4_sumspike += fc4_s
            # if fc1_v.shape[0] == 1:
                # NEURON_VTH_1 = NEURON_VTH_1.item()
                # NEURON_VTH_2 = NEURON_VTH_2.item()
                # NEURON_VTH_3 = NEURON_VTH_3.item()
                # NEURON_VTH_4 = NEURON_VTH_4.item()
                # with open(filename_1, 'a') as file_object1:
                #     file_object1.write(str(NEURON_VTH_1)+"\n")
                # with open(filename_2, 'a') as file_object2:
                #     file_object2.write(str(NEURON_VTH_2)+"\n")
                # with open(filename_3, 'a') as file_object3:
                #     file_object3.write(str(NEURON_VTH_3)+"\n")
                # with open(filename_4, 'a') as file_object4:
                #     file_object4.write(str(NEURON_VTH_4)+"\n")
                

        out = fc4_sumspike / self.batch_window
        
        

        return out, [fc1_u, fc1_v, fc1_s, fc2_u, fc2_v, fc2_s, fc3_u, fc3_v, fc3_s, fc4_u, fc4_v, fc4_s]


class ActorNetSpikingConv(nn.Module):
    """ Spiking Actor Network """
    def __init__(self, state_num, action_num, device, batch_window=50, hidden1=256, hidden2=256, hidden3=256):
        """

        :param state_num: number of states
        :param action_num: number of actions
        :param device: device used
        :param batch_window: window steps
        :param hidden1: hidden layer 1 dimension
        :param hidden2: hidden layer 2 dimension
        :param hidden3: hidden layer 3 dimension
        """
        super(ActorNetSpikingConv, self).__init__()
        self.state_num = state_num
        self.action_num = action_num
        self.device = device
        self.batch_window = batch_window
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.pseudo_spike = PseudoSpikeRect.apply
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(7*32, 256, bias=True)
        self.fc2 = nn.Linear(256, self.action_num, bias=True)

    def neuron_model(self, syn_func, pre_layer_output, current, volt, spike):
        """
        Neuron Model
        :param syn_func: synaptic function
        :param pre_layer_output: output from pre-synaptic layer
        :param current: current of last step
        :param volt: voltage of last step
        :param spike: spike of last step
        :return: current, volt, spike
        """
        current = current * NEURON_CDECAY + syn_func(pre_layer_output)
        volt = volt * NEURON_VDECAY * (1. - spike) + current
        spike = self.pseudo_spike(volt)
        return current, volt, spike

    def forward(self, x, batch_size):
        """

        :param x: state batch
        :param batch_size: size of batch
        :return: out
        """
        conv1_u = torch.zeros(batch_size, 32, 12, device=self.device)
        conv1_v = torch.zeros(batch_size, 32, 12, device=self.device)
        conv1_s = torch.zeros(batch_size, 32, 12, device=self.device)
        conv2_u = torch.zeros(batch_size, 32, 7, device=self.device)
        conv2_v = torch.zeros(batch_size, 32, 7, device=self.device)
        conv2_s = torch.zeros(batch_size, 32, 7, device=self.device)
        fc1_u = torch.zeros(batch_size, self.hidden3, device=self.device)
        fc1_v = torch.zeros(batch_size, self.hidden3, device=self.device)
        fc1_s = torch.zeros(batch_size, self.hidden3, device=self.device)
        fc2_u = torch.zeros(batch_size, self.action_num, device=self.device)
        fc2_v = torch.zeros(batch_size, self.action_num, device=self.device)
        fc2_s = torch.zeros(batch_size, self.action_num, device=self.device)
        fc2_sumspike = torch.zeros(batch_size, self.action_num, device=self.device)
        for step in range(self.batch_window):
            input_spike = x[:, :, step]
            conv1_u, conv1_v, conv1_s = self.neuron_model(self.conv1, input_spike, conv1_u, conv1_v, conv1_s)
            conv2_u, conv2_v, conv2_s = self.neuron_model(self.conv2, conv1_s, conv2_u, conv2_v, conv2_s)
            conv2_s = conv2_s.view(batch_size, -1)
            fc1_u, fc1_v, fc1_s = self.neuron_model(self.fc1, conv2_s, fc1_u, fc1_v, fc1_s)
            fc2_u, fc2_v, fc2_s = self.neuron_model(self.fc2, fc1_s, fc2_u, fc2_v, fc2_s)
            fc2_sumspike += fc2_s
        out = fc2_sumspike / self.batch_window
        return out
