import matplotlib.pyplot as plt
# from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib; matplotlib.use('Qt5Agg')

import pickle
import gzip
import numpy


import nest
import nest.raster_plot
import numpy as np


import scipy.io
import pickle
if not 'lif_psc_alpha_sl_ip' in nest.Models():
    nest.Install('LIF_SL_IPmodule')




Soma_Trained = np.zeros([4, 50, 324])
Recog_Trained = np.zeros([4,50, 37])
Noise_Trained = np.zeros([50, 11])
deg_list = [1, 9, 19, 28]

for simulations in range(0,50):

    for i in range(1, 38):  # 38 11
        # exec('file = "/Users/alex/Documents/IJASEIT Implementation of MNSD in NEST/Vision Pattern Decodificator/spikes_gabor18_noise/spikes_gabor_' + str(i) + '.pckl"')
        exec('file = "/Users/alex/Documents/IJASEIT Implementation of MNSD in NEST/Vision Pattern Decodificator/spikes_gabor18Final/spikes_gabor_' + str(i) + '.pckl"')
        with open(file, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            resultados = u.load()
        senders = resultados['senders'];
        times = np.round(resultados['times'] + np.random.randn(324) * 0.2, 1)  # We give some randomness to spike times. (Avoiding spikes to be equal)
        resultados_randomize = {'senders': senders, 'times': times}
        exec('file = "/Users/alex/Documents/IJASEIT Implementation of MNSD in NEST/Vision Pattern Decodificator/spikes_gabor_18_Final_rnd/spikes_gabor_18_randn02_SIMULATED_' + str(i) + '.pckl"')
        f = open(file, 'wb'); pickle.dump(resultados_randomize, f); f.close()

    for degree in range(0,4):
        SimTime = 200
        nest.ResetKernel()
        nest.SetKernelStatus({'resolution': 0.1})  # Time resolution set at 0.1 ms
        plt.close('all')

        exec('file = "/Users/alex/Documents/IJASEIT Implementation of MNSD in NEST/Vision Pattern Decodificator/spikes_gabor_18_Final_rnd/spikes_gabor_18_randn02_SIMULATED_' + str(deg_list[degree]) + '.pckl"')
        with open(file, 'rb') as f:
        #with open('/Users/alex/Documents/IJASEIT Implementation of MNSD in NEST/Vision Pattern Decodificator/spikes_gabor_18_Final_rnd/spikes_gabor_18_randn02_19.pckl', 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            resultados = u.load()
        senders = resultados['senders']; times = resultados['times'] # We give some randomness to spike times. (Avoiding spikes to be equal)

        GCells = len(senders); spikingGanglion = range(min(senders),max(senders)+1,1)
        inputs = nest.Create("spike_generator", GCells)
        for i in range(0,GCells):
            spike_time = times[senders == spikingGanglion[i]]
            nest.SetStatus([inputs[i]], {'spike_times': spike_time.tolist() })
            print(i, times[senders==spikingGanglion[i]])

        Ganglions = nest.Create('parrot_neuron', GCells)
        nest.Connect(inputs, Ganglions, 'one_to_one')

        Decoders = nest.Create('lif_psc_alpha_sl_ip', GCells, { 'tau_syn': 0.5, 'lambda': 0.0006, 'tau_decay': 0.02, 'tau': 50.0 })
        nest.Connect(Ganglions, Decoders, {'rule': 'one_to_one'}, {"weight": 3700.0}) # 3100.0  tuple(np.random.permutation(Decoders)) //// We can permutate the connections to biologically randomice pixel-detector.


        # We set the GID of neurons which modulate the Intrinsic Plasticity of the neuron.
        ###for i in range(0, GCells-2, 3):
               ################ nest.SetStatus([Decoders[i]], {'stimulator': [Decoders[i + 1]]});
              #############  nest.SetStatus([Decoders[i + 1]], {'stimulator': [Decoders[i], Decoders[i + 2]]});
             ###########   nest.SetStatus([Decoders[i + 2]], {'stimulator': [Decoders[i + 1]]});
          #########  # Connections between the 3 neurons
              #######  nest.Connect([Decoders[i]], [Decoders[i+1]], {"rule": "one_to_one"}, {"model": "stdp_synapse", 'delay': 0.1});
              #####  nest.Connect([Decoders[i+1]], [Decoders[i]], {"rule": "one_to_one"}, {"model": "stdp_synapse", 'delay': 0.1});
              ###  nest.Connect([Decoders[i+1]], [Decoders[i+2]], {"rule": "one_to_one"}, {"model": "stdp_synapse", 'delay': 0.1});
              ##  nest.Connect([Decoders[i+2]], [Decoders[i+1]], {"rule": "one_to_one"}, {"model": "stdp_synapse", 'delay': 0.1});

        # Target neuron. Connections are set in order to produce a target spike only in pattern detection.
        Targets = nest.Create('lif_psc_alpha_sl_ip', int(GCells/4), {'I_e': 0.0, "tau_minus": 10.0, 'tau_decay': 0.02})
        #j = 0
        #for i in range(0,GCells-2,3):
            #nest.Connect([Decoders[i], Decoders[i+1], Decoders[i+2]], [Targets[j]], {"rule": "all_to_all"}, {"weight": 283.5})  # 283.0 - 282.43 (mas o menos limitrofe) Probar 282.7
            #print(j, nest.GetConnections(target=[Targets[j]]))
            #j += 1

        k = 0
        for j in range(0,324,36):
            for i in range(0,18,2):
                #nest.SetStatus([Decoders[i + j]], {'stimulator': [Decoders[i + j + 1], Decoders[i + j + 18]]});
                #nest.SetStatus([Decoders[i + j + 2]], {'stimulator': [Decoders[i + j], Decoders[i + j + 19]]});
                #nest.SetStatus([Decoders[i + j + 18]], {'stimulator': [Decoders[i + j], Decoders[i + j + 19]]});
                #nest.SetStatus([Decoders[i + j + 19]], {'stimulator': [Decoders[i + j + 1], Decoders[i + j + 18]]});
                nest.SetStatus([Decoders[i + j]], {'stimulator': [Decoders[i + j + 1], Decoders[i + j + 18]]});#, Decoders[i+j+19]]});
                nest.SetStatus([Decoders[i + j + 1]], {'stimulator': [Decoders[i + j], Decoders[i + j + 19]]});#, Decoders[i+j+18]]});
                nest.SetStatus([Decoders[i + j + 18]], {'stimulator': [Decoders[i + j], Decoders[i + j + 19]]});#, Decoders[i+i+1]]});
                nest.SetStatus([Decoders[i + j + 19]], {'stimulator': [Decoders[i + j + 18], Decoders[i + j+1]]});#, Decoders[i+j]]});
                #nest.SetStatus([Decoders[i + j + 19]], {'stimulator': [Decoders[i + j + 1]]});#, Decoders[i + j + 18]]});
                # Connected in square
                nest.Connect([Decoders[i + j]], [Decoders[i + j + 1]], {"rule": "one_to_one"}, {"model": "stdp_synapse", 'delay': 0.1});
                nest.Connect([Decoders[i + j]], [Decoders[i + j + 18]], {"rule": "one_to_one"}, {"model": "stdp_synapse", 'delay': 0.1});
                nest.Connect([Decoders[i + j + 1]], [Decoders[i + j]], {"rule": "one_to_one"}, {"model": "stdp_synapse", 'delay': 0.1});
                nest.Connect([Decoders[i + j + 1]], [Decoders[i + j + 19]], {"rule": "one_to_one"}, {"model": "stdp_synapse", 'delay': 0.1});
                nest.Connect([Decoders[i + j + 18]], [Decoders[i + j]], {"rule": "one_to_one"}, {"model": "stdp_synapse", 'delay': 0.1});
                nest.Connect([Decoders[i + j + 18]], [Decoders[i + j + 19]], {"rule": "one_to_one"},{"model": "stdp_synapse", 'delay': 0.1});
                nest.Connect([Decoders[i + j + 19]], [Decoders[i + j + 1]], {"rule": "one_to_one"}, {"model": "stdp_synapse", 'delay': 0.1});
                nest.Connect([Decoders[i + j + 19]], [Decoders[i + j + 18]], {"rule": "one_to_one"}, {"model": "stdp_synapse", 'delay': 0.1});
                #nest.Connect([Decoders[i + j]], [Decoders[i + j + 19]], {"rule": "one_to_one"}, {"model": "stdp_synapse", 'delay': 0.1});
                #nest.Connect([Decoders[i + j + 19]], [Decoders[i + j]], {"rule": "one_to_one"}, {"model": "stdp_synapse", 'delay': 0.1});
                #nest.Connect([Decoders[i + j + 18]], [Decoders[i + j + 1]], {"rule": "one_to_one"}, {"model": "stdp_synapse", 'delay': 0.1});
                #nest.Connect([Decoders[i + j + 1]], [Decoders[i + j + 18]], {"rule": "one_to_one"}, {"model": "stdp_synapse", 'delay': 0.1});

                nest.Connect([Decoders[i+j], Decoders[i+j+1], Decoders[i+j+18], Decoders[i+j+19]], [Targets[k]], {"rule": "all_to_all"}, {"weight": 212.0})  # 283.0  212.2 212.0(+-10)
                k +=1


        # We create a Detector so that we can get spike times and raster plot
        Detector = nest.Create('spike_detector')
        nest.Connect(Ganglions, Detector)
        nest.Connect(Decoders, Detector)
        nest.Connect(Targets, Detector)
        # Recog = nest.Create('spike_detector')
        # nest.Connect(Targets, Recognitions)

        # We create a multim to record the Voltage potential of the membrane (V_m)
        # and the excitability parameter of Intrinsic Plasticity (soma_exc) /// DEBERIA PONERLO SOLO A LA ULTIMA PORQUE COGE UN PESO EXCESIVO
        Multimeter = nest.Create('multimeter', params={'withtime': True, 'record_from': ['V_m', 'soma_exc'], 'interval': 0.1})
        nest.Connect(Multimeter, Targets)
        nest.Connect(Multimeter, Decoders)


        ############# SIMULATION
        # We create a matrix to save the spike times of each trial
        iterations = 500; Recognitions = np.zeros([iterations,1])
        # times = np.zeros((iterations + 1, 97)); times[:, 97] = range(0, iterations + 1)
        for i in range(0,iterations):
            print(i)
            # First, we set the spike times
            for j in range(0,GCells):
                spike_time = (times[senders == spikingGanglion[j]])+200*i
                nest.SetStatus([inputs[j]], {'spike_times': spike_time.tolist()})
                # print(i,
            nest.SetStatus(Decoders, {"V_m": -70.0}); nest.SetStatus(Targets, {"V_m": -70.0})

            # if i == iterations-1:
            #     Recognitions = nest.Create('spike_detector')
            #     nest.Connect(Targets, Recognitions)
            Recog = nest.Create('spike_detector')   # Es una CAGADA crear cada uno, reprogramrlo o quitarlo !!!
            nest.Connect(Targets, Recog)

            nest.Simulate(200) # Simulate 1000 ms per trial (1s)


            print('Fires of Target Neurons in last trial: ', len(nest.GetStatus(Recog)[0]['events']['senders']), degree, simulations)
            if len(nest.GetStatus(Recog)[0]['events']['senders']) > 0:
                Recognitions[i] = len(nest.GetStatus(Recog)[0]['events']['senders']); print(Recognitions[i])
            disp = nest.GetStatus(Recog)[0]['events']['senders']  # Check las Target spikes to find what didnt spike


        ################## VISUALIZATION
        print(Recognitions)
        # Raster plot
        #nest.raster_plot.from_device(Detector)

        # V_m membrane potential
        ######################################events = nest.GetStatus(Multimeter)[0]['events']; eventos = events['V_m']
        # t = np.linspace(int(min(events['times'])), int(max(events['times']))+1, (int(max(events['times']))+1)-1);
        ##########################t = range(0, (iterations*2000-1),)
        ##############j = 0; v_1 = np.zeros([96,iterations*2000-1]); v_2 = np.zeros([96,iterations*2000-1]); v_3 = np.zeros([96,iterations*2000-1]); v_4 = np.zeros([96,iterations*2000-1]); v_5 = np.zeros([96,iterations*2000-1])
        ##########k = 0
        ##########for j in range(0,324,36):
            ##########for i in range(0,18,2):
                ###########  Vamo a poner lineas horizontales 96 hacia arriba, para ello multiplicamos por 100 cada una que añadamos
                ##########v_1[k,:] = events['V_m'][events['senders'] == Decoders[i+j]];
                ##########v_2[k,:] = events['V_m'][events['senders'] == Decoders[i+j+1]];
                ##########v_3[k,:] = events['V_m'][events['senders'] == Decoders[i+j+18]];
                ##########v_4[k, :] = events['V_m'][events['senders'] == Decoders[i + j + 19]];
                ##########v_5[k,:]= events['V_m'][events['senders'] == Targets[k]];
                ##########print('V_m calculating: ', k)
                ##########k +=1
            # se1 = events['soma_exc'][events['senders'] == 7] - 1;
            # se2 = events['soma_exc'][events['senders'] == 8] - 1;
            # se3 = events['soma_exc'][events['senders'] == 9] - 1;

        # IE_trained = np.zeros([288])
        # for i in range(0,289):
        #     # Extraemos el último Soma_exc value.
        #     IE_trained[i] = events['soma_exc'][events['times'] == iterations * 10000 - 2]


        #     ;v_2 = events['V_m'][events['senders']==8];v_3 = events['V_m'][events['senders']==9];v_4 = events['V_m'][events['senders']==10];
        # se1 = events['soma_exc'][events['senders']==7]-1;se2 = events['soma_exc'][events['senders']==8]-1;se3 = events['soma_exc'][events['senders']==9]-1;
        ############plt.figure(); plt.plot(t, v_1[59,:]); plt.plot(t, v_2[59,:]); plt.plot(t, v_3[59,:]); plt.plot(t, v_4[59,:]); plt.plot(t, v_5[59,:]); plt.ylabel('Membrane potential [mV]')
        #
        # Number of Target Spikes at final time
        print('Fires of Target Neurons in last trial: ', Recognitions[-1], degree, simulations)




        #####################  TEST ORIENTATIONS

        # In order to test other possible patterns, we swich off the Intrinsic Plasticity mode (std_mod: synaptic time dependent modificator)
        nest.SetStatus(Decoders, {'std_mod': False})

        Recognitions_test = np.zeros([37])

        Recogall = nest.Create('spike_detector')
        nest.Connect(Targets, Recogall)

        for i in range(1,38):

            exec('file = "/Users/alex/Documents/IJASEIT Implementation of MNSD in NEST/Vision Pattern Decodificator/spikes_gabor_18_Final_rnd/spikes_gabor_18_randn02_SIMULATED_'+str(i)+'.pckl"') #prueba18/spikes_gabor_
            with open(file, 'rb') as f:
                u = pickle._Unpickler(f)
                u.encoding = 'latin1'
                resultados = u.load()
            senders = resultados['senders']; times = resultados['times']  # We give some randomness to spike times. (Avoiding spikes to be equal)

        # First, we set the spike times
            for j in range(0,GCells):
                spike_time = ((times[senders == spikingGanglion[j]])+200*iterations)# + np.random.randn(1)*1, 1)
                nest.SetStatus([inputs[j]], {'spike_times': spike_time.tolist()})
                # print(i,
            nest.SetStatus(Decoders, {"V_m": -70.0}); nest.SetStatus(Targets, {"V_m": -70.0})

            # if i == iterations-1:
            #     Recognitions = nest.Create('spike_detector')
            #     nest.Connect(Targets, Recognitions)
            Recog = nest.Create('spike_detector')
            nest.Connect(Targets, Recog)

            nest.Simulate(200)

            if len(nest.GetStatus(Recog)[0]['events']['senders']) > 0:
                print('TEST with Gabor 5 (trained 19): ', len(nest.GetStatus(Recog)[0]['events']['senders']))
                Recognitions_test[i-1] = len(nest.GetStatus(Recog)[0]['events']['senders']); print('TEST Recognitions Gabor 5: ', Recognitions_test, 'TRAINED Gabor 19: ', Recognitions[-1])
            else:
                Recognitions_test[i-1] = 0
                print('No recognition...')
            iterations = iterations + 1

        print(Recognitions_test)

        # plt.figure(); plt.plot(Recognitions_test/1.08)
        #plt.figure(); t_MNSD = range(-90, 91, 5); t_theoretical = [-90, -67.5, -45, -22.5, 0, 22.5, 45, 67.5, 90]
        #plt.plot(t_MNSD, Recognitions_test*1.23456,'b+:',label='MNSD Detector')
        #theoretical =np.array([20.5, 20.75, 43.3, 64.4, 99.5, 62, 23.3, 21.5, 20.8])

        #plt.plot(t_theoretical, theoretical,'ro:',label='Theoretical')
        #plt.legend()
        # plt.title('Fig. 3 - Fit for Time Constant')
        #plt.xlabel('Orientation Tuning')
        #plt.ylabel('% Amplitude')
        #  Subirle el peso  283 para que cualquier randomn lo coja el detector con ese patr
        #plt.figure(); plt.plot(Recognitions)
        #nest.raster_plot.from_device(Recogall, hist = True, hist_binwidth=90.)

        #from scipy.ndimage.filters import gaussian_filter1d
        #ysmoothed = gaussian_filter1d(Recognitions_test, sigma=2);
        #plt.figure(); plt.plot(t_MNSD, ysmoothed * 1.23456)

        Recog_Trained[degree, simulations, :] = Recognitions_test; print(Recognitions_test)




        ########### Scatter SOMA_exc
        Soma_values = nest.GetStatus(Multimeter)[0]['events']['soma_exc'];
        last_values = Soma_values[-405:]; decoders_soma_exc = last_values[81:]
        #plt.figure(); plt.scatter( range(1,325), decoders_soma_exc)


        Soma_Trained[degree, simulations, :] = decoders_soma_exc

        if degree == 2:
            # %%    NOISE
            Noise_test = np.zeros([11]);
            Noise_test[0] = Recognitions_test[18];
            Recogg = Recog
            for i in range(1, 11):
                exec(
                    'file = "/Users/alex/Documents/IJASEIT Implementation of MNSD in NEST/Vision Pattern Decodificator/spikes_gabor_18_Final_rnd/spikes_gabor_18_noise_randn02_' + str(
                        i) + '.pckl"')
                with open(file, 'rb') as f:
                    u = pickle._Unpickler(f)
                    u.encoding = 'latin1'
                    resultados = u.load()
                senders = resultados['senders'];
                times = resultados['times']  # We give some randomness to spike times. (Avoiding spikes to be equal)

                # First, we set the spike times
                for j in range(0, GCells):
                    spike_time = ((times[senders == spikingGanglion[j]]) + 200 * iterations)  # + np.random.randn(1)*1, 1)
                    nest.SetStatus([inputs[j]], {'spike_times': spike_time.tolist()})
                    # print(i,
                nest.SetStatus(Decoders, {"V_m": -70.0});
                nest.SetStatus(Targets, {"V_m": -70.0})

                # if i == iterations-1:
                #     Recognitions = nest.Create('spike_detector')
                #     nest.Connect(Targets, Recognitions)
                Recog = nest.Create('spike_detector')
                nest.Connect(Targets, Recog)

                nest.Simulate(200)

                if len(nest.GetStatus(Recog)[0]['events']['senders']) > 0:
                    print('NOISE-TEST with Gabor (trained 19): ', len(nest.GetStatus(Recog)[0]['events']['senders']))
                    Noise_test[i] = len(nest.GetStatus(Recog)[0]['events']['senders']);
                    print('NOISE-TEST Recognitions Gabor: ', Noise_test, 'TRAINED Gabor 19: ', Recognitions[-1])
                else:
                    Noise_test[i] = 0
                    print('No recognition...')
                iterations = iterations + 1

            print(Noise_test);
            #Noise_test[0] = Recognitions_test[18]; Noise_transition = Noise_test
            #Noise_transition = [Recognitions_test[18], Noise_test]
            # plt.figure(); plt.plot(Noise_test/1.08)
            #  Subirle el peso  283 para que cualquier randomn lo coja el detector con ese patr
            #####plt.figure();
            #####n_MNSD = np.array(range(0, 100, 10));
            #####noise_theoretical = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]  # range(100,0, 10)
            #####plt.plot(n_MNSD, Noise_test / 0.81, 'b+:', label='MNSD Detector')
            #####plt.plot(n_MNSD, noise_theoretical, 'ro:', label='Theoretical')
            #####plt.legend()
            #####plt.xlabel('Noise')
            #####plt.ylabel('% Amplitude')

            Noise_Trained[simulations, :] = Noise_test

        print(degree)
    print(simulations)


# SAVE !!!!!!! GUARDAR DATOS
Simulated_data = {'Recog_Trained': Recog_Trained, 'Soma_Trained': Soma_Trained, 'Noise_Trained': Noise_Trained}
file = "/Users/alex/Documents/IJASEIT Implementation of MNSD in NEST/Vision Pattern Decodificator/Simulated_Data_Results_1OCT_Definitivo.pckl"
f = open(file, 'wb'); pickle.dump(Simulated_data, f); f.close()

#from scipy.ndimage.filters import gaussian_filter1d;
fig = plt.figure(); ax = fig.gca(); t_MNSD = range(-90, 91, 5); colors = ['r', 'y', 'b', 'g']; from scipy.interpolate import interp1d; xnew = np.linspace(-90, 90, num=500, endpoint=True)
for degree in range(0, 4):
    for simulations in range(0,50):
        eval('plt.plot(t_MNSD, Recog_Trained[degree, simulations, :]*1.23456, color ="' + str(colors[degree]) + '", linewidth = 0.3, alpha = 0.1)')
        #eval('plt.plot(t_MNSD, Recog_Trained[degree, simulations, :]*1.23456, color ="' + str(colors[degree]) + '", linewidth=0.5, alpha = 0.2)')
        #f2 = interp1d(t_MNSD, Recog_Trained[degree, simulations, :], kind='cubic');
        #eval('plt.plot(xnew, f2(xnew)*1.23456, color ="' + str(colors[degree]) + '", linewidth=0.5, alpha = 0.2)')
        #eval('plt.plot(t_MNSD, np.mean(Recog_Trained[degree, :, :] *1.23456, axis=0), c = "' + str(colors[degree]) + '")')

    #exec('ysmoothed = gaussian_filter1d(np.mean(Recog_Trained[degree,:,:], axis=0), sigma=1); plt.plot(t_MNSD, ysmoothed*1.23456, c = "' + str(colors[degree]) + '")')
    #f2 = interp1d(t_MNSD, np.mean(Recog_Trained[degree,:,:], axis=0), kind='cubic');
    #eval('plt.plot(xnew, f2(xnew) * 1.23456, c="' + str(colors[degree]) + '")')
    eval('plt.plot(t_MNSD, np.mean(Recog_Trained[degree,:,:], axis=0) * 1.23456, c="' + str(colors[degree]) + '")')
    # Dibujar zona de error:
    maxline = np.zeros([37]); minline =  np.zeros([37]);
    for i in range(0,37):
        maxline[i]= max(Recog_Trained[degree, :, i]); minline[i]= min(Recog_Trained[degree, :, i]);
    eval('plt.fill_between(t_MNSD, maxline*1.23456, minline*1.23456, color="' + str(colors[degree]) + '", alpha = 0.2)')
    ######for i in range(0,37):
        ######for j in range(0,10):
            ######max_value[j, i] = max(Recog_Trained[degree, j, i]); min_value[j,i] = min(Recog_Trained[degree, j, i]);
    ######plt.fill_between(t_MNSD,

fig.savefig('Orientations_tuning.pdf', format='pdf')


from scipy.ndimage.filters import gaussian_filter1d;
plt.figure(); t_Noise = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]; colors = ['r', 'y', 'b', 'g']
for simulations in range(0,50):
    plt.plot(t_Noise, Noise_Trained[simulations, :]*1.23456, color ="b", linewidth=0.5, alpha = 0.2)
    #exec('ysmoothed = gaussian_filter1d(np.mean(Noise_Trained[:,:], axis=0), sigma=1); plt.plot(t_Noise, ysmoothed*1.23456, c = "' + str(colors[degree]) + '")')
plt.plot(t_Noise, np.mean(Noise_Trained[:,:], axis=0) * 1.23456, c="b")
maxline = np.zeros([11]); minline =  np.zeros([11]);
for i in range(0, 11):
    maxline[i] = max(Noise_Trained[ :, i]);
    minline[i] = min(Noise_Trained[ :, i]);
plt.fill_between(t_Noise, maxline*1.23456, minline*1.23456, color="b", alpha = 0.2)


plt.savefig('Noise_recognition.pdf', format='pdf')


plt.figure(); medias = np.zeros([4,1]); desviaciones = np.zeros([4,1]);
for degree in range(0, 4):
    #for simulations in range(0, 10):
        #eval('plt.scatter(range(1,325), Soma_Trained[degree,simulations, :], c = "' + str(colors[degree]) + '", s = 1)')
    # eval('plt.plot(t_MNSD, np.mean(Recog_Trained[degree, :, :] *1.23456, axis=0), c = "' + str(colors[degree]) + '")')
    #j = 0; pointsdata = np.zeros([1625])
    #for i in range(0,324):
     #   pointsdata[j] = 0;pointsdata[j+1] = 0;pointsdata[j+2] = 0;pointsdata[j+3] = 0;
      #  pointsdata[j+4] = np.mean(Soma_Trained[degree, :, i], axis=0);
       # j = j+5;

    #eval('plt.scatter(range(1,1626), pointsdata, c = "' + str(colors[degree]) + '", s = 12)')
    eval('plt.scatter(range(1,325), np.mean(Soma_Trained[degree, :, :], axis=0), c = "' + str(colors[degree]) + '", s = 12)')
    eval('plt.plot(range(1,325), np.mean(Soma_Trained[degree, :, :], axis=0), c = "' + str(colors[degree]) + '", linewidth=0.5)')
    #desviaciones[degree] = np.mean(np.std(Soma_Trained[degree,:, :], axis=0))

plt.savefig('IE_values.pdf', format='pdf')



#plt.figure(); randomizaciones = np.random.randn(4,1000)
#for i in range(0,4):
  #  randomizaciones[i, :] = randomizaciones[i, :] * desviaciones[i] + medias[i]
 #   plt.hist(randomizaciones[i,:])

################







#from scipy.ndimage.filters import gaussian_filter1d;
#ysmoothed = gaussian_filter1d(Noise_test, sigma=2);
#plt.figure(); plt.plot(n_MNSD, ysmoothed * 1.23456)

#nest.raster_plot.from_device(Recogg)
#plt.figure(); plt.plot(Recognitions)




import pickle;
file = "/Users/alex/Documents/IJASEIT Implementation of MNSD in NEST/Vision Pattern Decodificator/Simulated_Data_Results_1OCT_Definitivo.pckl"
f = open(file, 'rb'); data = pickle.load(f)
Soma_data = data['Soma_Trained']

plt.figure(); colors = ['Reds', 'YlOrBr', 'Blues', 'Greens']
Soma_matrix = np.zeros([18,18,4])
for degree in range(0,4):
    Soma_vector = np.mean(Soma_data[degree, :, :], axis=0)
    for i in range(0,18):
        Soma_matrix[i,:, degree] = Soma_vector[18*i:18*i+18]
    plt.subplot(2,2,degree+1)
    plt.imshow(Soma_matrix[:,:,degree], cmap = colors[degree], vmin=0.8, vmax=1.3)
    plt.colorbar()

plt.savefig('IE_images.pdf', format='pdf')

