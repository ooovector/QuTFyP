# Unit tests 
import unittest
import numpy as np
import QuTFyP
import tensorflow as tf
import math

class TransmonQEDTestCase(unittest.TestCase):
    def setUp(self):
        self.qed = QuTFyP.TransmonQED()
        self.qed.d = [2, 4]
        self.qed.initialize_operators()
        self.qed.ntraj = 2
        self.qed.rdtype_np = np.float32
        self.qed.cdtype_np = np.complex64
        self.qed.rdtype = tf.float32
        self.qed.cdtype = tf.complex64
    def tearDown(self):
        del self.qed
    def testMultipliers(self):
        #n = self.qed.mutipliers_sqr
        with tf.Session() as sess:
            n_check = [[], []];
            n_check[0] = np.asarray([[[0]], [[1]]], dtype=self.qed.cdtype_np)
            n_check[1] = np.asarray([[[0], [1], [2], [3]]], dtype=self.qed.cdtype_np)
            n_check = np.asarray(n_check)
            #sqrt_n_check = np.sqrt(n_check)
            n_str = np.array2string(np.asarray(self.qed.multipliers_sqr))
            n_check_str = np.array2string(n_check)
            #sqrt_n_str = np.array2string(self.qed.multipliers)
            #sqrt_n_check_str = np.array2string(sqrt_n_check)
            assert np.all([np.all(n_check[i] == sess.run(self.qed.multipliers_sqr[i])) for i in range(len(n_check))]), \
            '''TransmonQED.multipliers_sqr failed check against predefined.
TransmonQED.multipliers_sqr:'''+n_str+'''
Check value:'''+n_check_str

    def testOpVec(self):
        with tf.Session() as sess:
            # start with 0-th state for qubit and 2-nd state for resonator
            gi = [[0,2,i] for i in range(self.qed.ntraj)]  # A list of coordinates to update.
            gv = [1.0]*self.qed.ntraj  # A list of values corresponding to the respective
            psi = self.qed.pure_state(gi, gv)
            sess.run(psi.initializer)
            am_psi = sess.run(self.qed.am(psi, ax=1))
            ap_psi = sess.run(self.qed.ap(psi, ax=1))
    
            am_gi = [[0,1,i] for i in range(self.qed.ntraj)]  # A list of coordinates to update.
            am_gv = [np.sqrt(2.0)]*self.qed.ntraj  # A list of values corresponding to the respective
            am_psi_check = self.qed.pure_state(am_gi, am_gv)
            sess.run(am_psi_check.initializer)
            assert np.all(am_psi == sess.run(am_psi_check)), '''TransmonQED.am failed check against predefined.
TransmonQED.am:'''+np.array2string(am_psi)+'''
Check value:'''+np.array2string(sess.run(am_psi_check))
            
            ap_gi = [[0,3,i] for i in range(self.qed.ntraj)]  # A list of coordinates to update.
            ap_gv = [np.sqrt(3.0)]*self.qed.ntraj  # A list of values corresponding to the respective
            ap_psi_check = self.qed.pure_state(am_gi, am_gv)
            sess.run(ap_psi_check.initializer)
            assert np.all(am_psi == sess.run(ap_psi_check)), '''TransmonQED.ap failed check against predefined.
TransmonQED.ap:'''+np.array2string(ap_psi)+'''
Check value:'''+np.array2string(sess.run(ap_psi_check))

    def testHintVec(self):
        self.qed.anharmonicities = [-0.2, 0.0]
        self.qed.frequencies = [6.0, 9.5] # in GHz. Transmon is 6.0, resonator is 9.5.
        self.qed.couplings = [[0, 0.1],[0.1, 0]] # coupling constants between different degrees of freedom
        # decoherences = [[0.01, 0.000, 0.00], [0.00, 0.000, 0.000]] # relaxation, thermal excitation, pure dephasing rate triples
        gi = [[0,1,i] for i in range(self.qed.ntraj)]  # A list of coordinates to update.
        gv = [1.0]*self.qed.ntraj  # A list of values corresponding to the respective
        psi = self.qed.pure_state(gi, gv)
        
        with tf.Session() as sess:
            sess.run(psi.initializer)
            Hint = sess.run(self.qed.Hint(psi, 0))
            
        Hint_check = np.asarray([[[0, 0], [0, 0], [0, 0], [0, 0]], 
                                 [[0.1, 0.1], [0, 0], [0.1*math.sqrt(2), 0.1*math.sqrt(2)], [0, 0]]], dtype=self.qed.cdtype_np)
        #print (np.sum(np.abs(Hint_check-Hint)))
        assert np.sum(np.abs(Hint_check-Hint))<np.sum(np.abs(Hint))*1e-6, '''TransmonQED.Hint failed check against predefined.
TransmonQED.Hint:'''+np.array2string(Hint)+'''
Check value:'''+np.array2string(Hint_check)
        
    def testHintMat(self):
        self.qed.anharmonicities = [-0.2, 0.0]
        self.qed.frequencies = [6.0, 9.5] # in GHz. Transmon is 6.0, resonator is 9.5.
        self.qed.couplings = [[0, 0.1],[0.1, 0]] # coupling constants between different degrees of freedom
        # decoherences = [[0.01, 0.000, 0.00], [0.00, 0.000, 0.000]] # relaxation, thermal excitation, pure dephasing rate triples
        gi = [[0,1,i] for i in range(self.qed.ntraj)]  # A list of coordinates to update.
        gv = [1.0]*self.qed.ntraj  # A list of values corresponding to the respective
        
        # check state matrix against state vector
        psi = self.qed.pure_state(gi, gv)
        rho = tf.einsum('ijn,kln->ijkln', psi, tf.conj(psi))
        Hint_rho = 1j*self.qed.Hint_d2(rho, 0)
        Hint_psi = 1j*self.qed.Hint(psi, 0)
        Hint_rho_check = tf.einsum('ijn,kln->ijkln', Hint_psi, tf.conj(psi))+tf.einsum('ijn,kln->ijkln', psi, tf.conj(Hint_psi))
        
        with tf.Session() as sess:
            sess.run(psi.initializer)
            Hint_rho_result = sess.run(Hint_rho)
            Hint_rho_check_result = sess.run(Hint_rho_check)
            
        #print (np.sum(np.abs(Hint_check-Hint)))
        
        fail_mask = np.abs(Hint_rho_result-Hint_rho_check_result)>np.abs(Hint_rho_check_result)*1e-6
        pass_mask = np.logical_and(np.abs(Hint_rho_check_result)*1e-6, 1-fail_mask)
        indices = np.indices(fail_mask.shape)
        def mask_to_str(mask):
            mask_indices = np.asarray([i[mask] for i in indices]).T
            fail_indices_str = [', '.join([str(j) for j in i]) for i in mask_indices.tolist()]
            fail_Hint_rho = Hint_rho_result[mask]
            fail_Hint_rho_check = Hint_rho_check_result[mask]
            return '\n'.join('\t'.join([str(f) for f in fail]) for fail in zip(fail_indices_str, fail_Hint_rho, fail_Hint_rho_check))
            #print (np.abs(Hint_rho_result-Hint_rho_check_result))
        fail_list_str = mask_to_str(fail_mask)
        pass_list_str = mask_to_str(pass_mask)
        assert np.sum(np.abs(Hint_rho_result-Hint_rho_check_result))<np.sum(np.abs(Hint_rho_check_result))*1e-6, '''TransmonQED.Hint_d2 failed check against TransmonQED.Hint.
Indeces, TransmonQED.Hint_d2, Check value:
'''+fail_list_str+'Passed check:\nIndeces, TransmonQED.Hint_d2, Check value:\n'+pass_list_str

    def testH(self):
        # test hermitian part of liouvillian and hamiltonian
        self.qed.anharmonicities = [-0.2, 0.0]
        self.qed.frequencies = [6.0, 9.5] # in GHz. Transmon is 6.0, resonator is 9.5.
        self.qed.couplings = [[0, 0.1],[0.1, 0]] # coupling constants between different degrees of freedom
        self.qed.decoherences = [] # relaxation, thermal excitation, pure dephasing rate triples
        gi = [[0,1,i] for i in range(self.qed.ntraj)]  # A list of coordinates to update.
        gv = [1.0]*self.qed.ntraj  # A list of values corresponding to the respective
        
        f2 = np.linspace(6, 7, 2)
        a2 = np.linspace(1, 2, 2)
        controls = {'ex_x':{'carrier':f2,  'phi':0,        'coupling':[{}, {'y':-a2}], 'envelope': lambda x: tf.sin(x)},
                    'ro_x':{'carrier':9.5, 'phi':0,        'coupling':[{'y':-1}, {}], 'envelope': lambda x: tf.sin(x)},
                    'ex_y':{'carrier':f2,  'phi':np.pi/2., 'coupling':[{}, {'x': a2}], 'envelope': lambda x: tf.sin(x)},
                    'ro_y':{'carrier':9.5, 'phi':np.pi/2., 'coupling':[{'x': 1}, {}], 'envelope': lambda x: tf.sin(x)}}
        self.qed.controls = controls
        
        # check state matrix against state vector
        # 1. test interaction hamiltonian at random t
        t = np.random.rand(10)*10 # random t within the first 10 time units
        function_pairs = ((self.qed.Hint, self.qed.Hint_d2, 'TransmonQED.Hint'), 
                          (self.qed.V, self.qed.V_d2, 'TransmonQED.V'))

        psi = self.qed.pure_state(gi, gv)
        rho = tf.einsum('ijn,kln->ijkln', psi, tf.conj(psi))        
        
        for f_pair in function_pairs: 
            for _t in t:
                H_rho = 1j*f_pair[1](rho, _t)
                H_psi = 1j*f_pair[0](psi, _t)
                H_rho_check = tf.einsum('ijn,kln->ijkln', H_psi, tf.conj(psi))+tf.einsum('ijn,kln->ijkln', psi, tf.conj(H_psi))
                with tf.Session() as sess:
                    sess.run(psi.initializer)
                    H_rho_result = sess.run(H_rho)
                    H_rho_check_result = sess.run(H_rho_check)

                    fail_mask = np.abs(H_rho_result-H_rho_check_result)>np.abs(H_rho_check_result)*1e-6
                    pass_mask = np.logical_and(np.abs(H_rho_check_result)*1e-6, 1-fail_mask)
                    indices = np.indices(fail_mask.shape)
                    def mask_to_str(mask):
                        mask_indices = np.asarray([i[mask] for i in indices]).T
                        fail_indices_str = [', '.join([str(j) for j in i]) for i in mask_indices.tolist()]
                        fail_Hint_rho = H_rho_result[mask]
                        fail_Hint_rho_check = H_rho_check_result[mask]
                        return '\n'.join('\t'.join([str(f) for f in fail]) for fail in zip(fail_indices_str, fail_Hint_rho, fail_Hint_rho_check))
                        #print (np.abs(Hint_rho_result-Hint_rho_check_result))
                fail_list_str = mask_to_str(fail_mask)
                pass_list_str = mask_to_str(pass_mask)
                assert np.sum(np.abs(H_rho_result-H_rho_check_result))<np.sum(np.abs(H_rho_check_result))*1e-6, '''{0}_d2 failed check against {0} @ random time {1}.
Indeces, {0}_d2, {0} value:
'''.format(f_pair[2], _t)+fail_list_str+'Passed check:\nIndeces, {0}_d2, {0}:\n'.format(f_pair[2], _t)+pass_list_str

    def testConditionalDecay(self):
         # test hermitian part of liouvillian and hamiltonian
        #self.qed.ntraj = 20
        self.qed.ntraj = 400
        self.qed.anharmonicities = [0.0, 0.0]
        self.qed.frequencies = [6.0, 9.5] # in GHz. Transmon is 6.0, resonator is 9.5.
        self.qed.couplings = [[0, 0],[0, 0]] # coupling constants between different degrees of freedom
        self.qed.decoherences = {'qubit_decay':{'measurement_type':'homodyne',
                                  'coupling_type':'a',
                                  'rate':2.0,
                                  'subsystem_id':0,
                                  'record':True,
                                  'resample':True,
                                  'window_start':0.,
                                  'window_stop':2.}} # relaxation, thermal excitation, pure dephasing rate triples
        self.qed.initialize_operators()
        self.qed.controls = {}
        self.qed.minibatch = 256 # number of time steps with a single noise sample and without resampling

        gi = [[1,0,i] for i in range(self.qed.ntraj)]  # A list of coordinates to update.
        gv = [1.0]*self.qed.ntraj  # A list of values corresponding to the respective
                # coordinate in indices.
        self.qed.set_initial_pure_state(gi, gv)
        psi = self.qed.initial_state_vec
        #psi = self.qed.initial_state_vec
        #rho = tf.einsum('ijn,kln->ijkln', psi, tf.conj(psi))        
        
        # unitary evolution 
        self.qed.simulation_time = 1.0
        self.qed.dt = 0.005
        self.qed.minibatch = 1

        quantum_noise = tf.random_normal([self.qed.ntraj, 1, self.qed.minibatch], 
                                         mean=0.0, 
                                         stddev=1.0, 
                                         dtype=self.qed.rdtype, 
                                         seed=None, 
                                         name='quantum_noise')
        random_hamiltonian, measurement = self.qed.homodyne_conditional(psi, 0, quantum_noise)
        with tf.Session() as sess:
            sess.run(psi.initializer)
            print (sess.run(measurement))

    def testUnitaryEvolution(self):
        # test hermitian part of liouvillian and hamiltonian
        self.qed.anharmonicities = [-0.2, 0.0]
        self.qed.frequencies = [6.0, 9.5] # in GHz. Transmon is 6.0, resonator is 9.5.
        self.qed.couplings = [[0, 0.1],[0.1, 0]] # coupling constants between different degrees of freedom
        self.qed.decoherences = {} # relaxation, thermal excitation, pure dephasing rate triples
        gi = [[0,1,i] for i in range(self.qed.ntraj)]  # A list of coordinates to update.
        gv = [1.0]*self.qed.ntraj  # A list of values corresponding to the respective
        
        self.qed.expectations = { #'qubit_x': {'resample':True,
                                  #'window_start':5., 
                    #              'window_stop':7., 
                    #'observable_vec': lambda x: self.qed.observable(tf.real(tf.conj(x)*self.qed.am(x, ax=0))),
                    #'observable_mat': lambda x: self.qed.observable(tf.real(self.am_d2(x, ax=0)), mode='mat') },
                                  'qubit_z': {'resample':False,
                    'observable_vec': lambda x: self.qed.observable(tf.real(tf.conj(x)*self.qed.multipliers_sqr_real[0])),
                    'observable_mat': lambda x: self.qed.observable(tf.real(self.qed.multipliers_sqr_real[0]*x), mode='mat') } }
        
        f2 = np.linspace(6, 7, 2)
        a2 = np.linspace(1, 2, 2)
        controls = {'ex_x':{'carrier':f2,  'phi':0,        'coupling':[{}, {'y':-a2}], 'envelope': lambda x: tf.sin(x)},
                    'ro_x':{'carrier':9.5, 'phi':0,        'coupling':[{'y':-1}, {}], 'envelope': lambda x: tf.sin(x)},
                    'ex_y':{'carrier':f2,  'phi':np.pi/2., 'coupling':[{}, {'x': a2}], 'envelope': lambda x: tf.sin(x)},
                    'ro_y':{'carrier':9.5, 'phi':np.pi/2., 'coupling':[{'x': 1}, {}], 'envelope': lambda x: tf.sin(x)}}
        self.qed.controls = controls


        gi = [[1,0,i] for i in range(self.qed.ntraj)]  # A list of coordinates to update.
        gv = [1.0]*self.qed.ntraj  # A list of values corresponding to the respective
                # coordinate in indices.

        self.qed.set_initial_pure_state(gi, gv)
        psi = self.qed.initial_state_vec
        #rho = tf.einsum('ijn,kln->ijkln', psi, tf.conj(psi))        
        
        # unitary evolution 
        self.qed.simulation_time = 0.2
        self.qed.dt = 0.001
        expectations_vec = self.qed.run('vec')
        expectations_mat = self.qed.run('mat_pure')
        expect_diff = np.sum(np.abs(expectations_vec-expectations_mat))
        expect_avg = np.sum(np.abs(expectations_vec))
        print('Expectation rel error: {}'.format(expect_diff/expect_avg))
        #print (expectations_vec)
        
    def testExpectations(self):
        # test hermitian part of liouvillian and hamiltonian
        self.qed.anharmonicities = [-0.2, 0.0]
        self.qed.frequencies = [6.0, 9.5] # in GHz. Transmon is 6.0, resonator is 9.5.
        self.qed.couplings = [[0, 0.1],[0.1, 0]] # coupling constants between different degrees of freedom
        self.qed.decoherences = {} # relaxation, thermal excitation, pure dephasing rate triples
        gi = [[0,1,i] for i in range(self.qed.ntraj)]  # A list of coordinates to update.
        gv = [1.0]*self.qed.ntraj  # A list of values corresponding to the respective
        
        self.qed.expectations = { #'qubit_x': {'resample':True,
                                  #'window_start':5., 
                    #              'window_stop':7., 
                    #'observable_vec': lambda x: self.qed.observable(tf.real(tf.conj(x)*self.qed.am(x, ax=0))),
                    #'observable_mat': lambda x: self.qed.observable(tf.real(self.am_d2(x, ax=0)), mode='mat') },
                                  'qubit_z': {'resample':False,
                    'observable_vec': lambda x: self.qed.observable(tf.real(tf.conj(x)*self.qed.multipliers_sqr_real[0])),
                    'observable_mat': lambda x: self.qed.observable(tf.real(self.qed.multipliers_sqr_real[0]*x), mode='mat') } }
        
        f2 = np.linspace(6, 7, 2)
        a2 = np.linspace(1, 2, 2)
        controls = {'ex_x':{'carrier':f2,  'phi':0,        'coupling':[{}, {'y':-a2}], 'envelope': lambda x: tf.sin(x)},
                    'ro_x':{'carrier':9.5, 'phi':0,        'coupling':[{'y':-1}, {}], 'envelope': lambda x: tf.sin(x)},
                    'ex_y':{'carrier':f2,  'phi':np.pi/2., 'coupling':[{}, {'x': a2}], 'envelope': lambda x: tf.sin(x)},
                    'ro_y':{'carrier':9.5, 'phi':np.pi/2., 'coupling':[{'x': 1}, {}], 'envelope': lambda x: tf.sin(x)}}
        self.qed.controls = controls


        gi = [[1,0,i] for i in range(self.qed.ntraj)]  # A list of coordinates to update.
        gv = [1.0]*self.qed.ntraj  # A list of values corresponding to the respective
                # coordinate in indices.

        self.qed.set_initial_pure_state(gi, gv)
        psi = self.qed.initial_state_vec
        #rho = tf.einsum('ijn,kln->ijkln', psi, tf.conj(psi))        
        
        # unitary evolution 
        self.qed.simulation_time = 0.2
        self.qed.dt = 0.001
        with tf.Session() as sess:
            sess.run(psi.initializer)
            print(sess.run(self.qed.calc_expectations(psi)))
        #expectations_vec = self.qed.run('vec')
        #expectations_mat = self.qed.run('mat_pure')
        #expect_diff = np.sum(np.abs(expectations_vec-expectations_mat))
        #expect_avg = np.sum(np.abs(expectations_vec))
        #print('Expectation rel error: {}'.format(expect_diff/expect_avg))
        #print (expectations_vec)

    def testDecay(self):
        # test hermitian part of liouvillian and hamiltonian
        #self.qed.ntraj = 1
        self.qed.ntraj = 100
        self.qed.anharmonicities = [0.0, 0.0]
        self.qed.frequencies = [6.0, 9.5] # in GHz. Transmon is 6.0, resonator is 9.5.
        self.qed.couplings = [[0, 0],[0, 0]] # coupling constants between different degrees of freedom
        self.qed.decoherences = {'qubit_decay':{'measurement_type':'photon-counting',
                                  'coupling_type':'a',
                                  'rate':2.0,
                                  'subsystem_id':0,
                                  'record':True,
                                  'resample':True,
                                  'window_start':0.,
                                  'window_stop':2.}}
        self.qed.initialize_operators()
        self.qed.controls = {}
        self.qed.minibatch = 50 # number of time steps with a single noise sample and without resampling
        self.qed.expectations = { 'qubit_x': {'unmix':False,
                                  'window_start':5., 
                                  'window_stop':7., 
                    'observable_vec': lambda x: self.qed.observable(tf.real(tf.conj(x)*self.qed.am(x, ax=0))),
                    'observable_mat': lambda x: self.qed.observable(tf.real(self.qed.am_d2(x, ax=0)), mode='mat') },
                                  'qubit_y': {'unmix':False,
                                  'window_start':5., 
                                  'window_stop':7., 
                    'observable_vec': lambda x: self.qed.observable(tf.imag(tf.conj(x)*self.qed.am(x, ax=0))),
                    'observable_mat': lambda x: self.qed.observable(tf.imag(self.qed.am_d2(x, ax=0)), mode='mat') },
                                  'qubit_z': {'unmix':False,
                                  'window_start':0., 
                                  'window_stop':2.,
                    'observable_vec': lambda x: self.qed.observable(tf.real(tf.conj(x)*x*self.qed.multipliers_sqr_real[0])),
                    'observable_mat': lambda x: self.qed.observable(tf.real(self.qed.multipliers_sqr_real[0]*x), mode='mat') },
                                  'resonator_x': {'unmix':False,
                                  'window_start':8.5,
                                  'window_stop':10.5, 
                    'observable_vec': lambda x: self.qed.observable(tf.real(tf.conj(x)*self.qed.am(x, ax=1))),
                    'observable_mat': lambda x: self.qed.observable(tf.real(self.qed.am_d2(x, ax=1)), mode='mat') },
                                  'resonator_y': {'unmix':False,
                                  'window_start':8.5,
                                  'window_stop':10.5, 
                    'observable_vec': lambda x: self.qed.observable(tf.imag(tf.conj(x)*self.qed.am(x, ax=1))),
                    'observable_mat': lambda x: self.qed.observable(tf.imag(self.qed.am_d2(x, ax=1)), mode='mat') },
                                  'resonator_z': {'unmix':False,
                                  'window_start':0,
                                  'window_stop':2,
                    'observable_vec': lambda x: self.qed.observable(tf.real(tf.conj(x)*x*self.qed.multipliers_sqr_real[1])),
                    'observable_mat': lambda x: self.qed.observable(tf.real(self.qed.multipliers_sqr_real[1]*x), mode='mat') }}

        gi = [[1,0,i] for i in range(self.qed.ntraj)]  # A list of coordinates to update.
        gv = [1.0]*self.qed.ntraj  # A list of values corresponding to the respective
                # coordinate in indices.
        self.qed.set_initial_pure_state(gi, gv)
        #psi = self.qed.initial_state_vec
        #rho = tf.einsum('ijn,kln->ijkln', psi, tf.conj(psi))        
        
        # unitary evolution 
        self.qed.simulation_time = 2.0
        self.qed.dt = 0.01
        expectations_vec, measurements_vec = self.qed.run('vec')
        expectations_mat = self.qed.run('mat_pure')
        self.qed.decoherences['qubit_decay']['measurement_type'] = 'homodyne'
        self.qed.decoherences['qubit_decay']['noise_spectral_density'] = np.ones((1,), dtype=self.qed.rdtype_np)
        expectations_vec_homodyne, measurements_vec_homodyne = self.qed.run('vec')

        from matplotlib.pyplot import plot as plot
        from matplotlib.pyplot import legend as legend
        #print (expectations_vec['qubit_z'])
        plot(np.mean(expectations_vec['qubit_z'], axis=1).T, label='Jumps <n>')
        plot(np.mean(expectations_mat['qubit_z'], axis=1).T, label='ME <n>')
        plot(np.mean(expectations_vec_homodyne['qubit_z'], axis=1).T, label='Homodyne <n>')
        plot(np.mean(expectations_vec['qubit_x'], axis=1).T, label='Jumps Re{<a>}')
        plot(np.mean(expectations_mat['qubit_x'], axis=1).T, label='ME Re{<a>}')
        plot(np.mean(expectations_vec_homodyne['qubit_x'], axis=1).T, label='Homodyne Re{<a>}')

        legend()

        self.tearDown()
        self.setUp()
        
    def testDownSampling(self):
        # test hermitian part of liouvillian and hamiltonian
        #self.qed.ntraj = 1
        self.qed.ntraj = 100
        self.qed.anharmonicities = [0.0, 0.0]
        self.qed.frequencies = [6.0, 9.5] # in GHz. Transmon is 6.0, resonator is 9.5.
        self.qed.couplings = [[0, 0],[0, 0]] # coupling constants between different degrees of freedom
        self.qed.decoherences = {'qubit_decay':{'measurement_type':'photon-counting',
                                  'coupling_type':'a',
                                  'rate':2.0,
                                  'subsystem_id':0,
                                  'record':True,
                                  'unmix':False,
                                  'unmix_quad'='I'
                                  'unmix_reference':0.,
                                  'sample_rate':2.}}
        self.qed.initialize_operators()
        self.qed.controls = {}
        self.qed.minibatch = 50 # number of time steps with a single noise sample and without resampling
        self.qed.expectations = { 'qubit_x': {'unmix':True,
                                  'unmix_quad':5., 
                                  'unmix_reference':0., 
                                  'sample_rate':1.,
                    'observable_vec': lambda x: self.qed.observable(tf.real(tf.conj(x)*self.qed.am(x, ax=0))),
                    'observable_mat': lambda x: self.qed.observable(tf.real(self.qed.am_d2(x, ax=0)), mode='mat') },
                                  'qubit_y': {'unmix':True,
                                  'window_start':5., 
                                  'window_stop':7., 
                    'observable_vec': lambda x: self.qed.observable(tf.imag(tf.conj(x)*self.qed.am(x, ax=0))),
                    'observable_mat': lambda x: self.qed.observable(tf.imag(self.qed.am_d2(x, ax=0)), mode='mat') },
                                  'qubit_z': {'unmix':True,
                                  'window_start':0., 
                                  'window_stop':2.,
                    'observable_vec': lambda x: self.qed.observable(tf.real(tf.conj(x)*x*self.qed.multipliers_sqr_real[0])),
                    'observable_mat': lambda x: self.qed.observable(tf.real(self.qed.multipliers_sqr_real[0]*x), mode='mat') },
                                  'qubit_z_high': {'unmix':False,
                                  'window_start':0., 
                                  'window_stop':2.,
                    'observable_vec': lambda x: self.qed.observable(tf.real(tf.conj(x)*x*self.qed.multipliers_sqr_real[0])),
                    'observable_mat': lambda x: self.qed.observable(tf.real(self.qed.multipliers_sqr_real[0]*x), mode='mat') },
                                  'resonator_x': {'unmix':True,
                                  'window_start':8.5,
                                  'window_stop':10.5, 
                    'observable_vec': lambda x: self.qed.observable(tf.real(tf.conj(x)*self.qed.am(x, ax=1))),
                    'observable_mat': lambda x: self.qed.observable(tf.real(self.qed.am_d2(x, ax=1)), mode='mat') },
                                  'resonator_y': {'unmix':True,
                                  'window_start':8.5,
                                  'window_stop':10.5, 
                    'observable_vec': lambda x: self.qed.observable(tf.imag(tf.conj(x)*self.qed.am(x, ax=1))),
                    'observable_mat': lambda x: self.qed.observable(tf.imag(self.qed.am_d2(x, ax=1)), mode='mat') },
                                  'resonator_z': {'unmix':True,
                                  'window_start':0,
                                  'window_stop':2,
                    'observable_vec': lambda x: self.qed.observable(tf.real(tf.conj(x)*x*self.qed.multipliers_sqr_real[1])),
                    'observable_mat': lambda x: self.qed.observable(tf.real(self.qed.multipliers_sqr_real[1]*x), mode='mat') }}

        gi = [[1,0,i] for i in range(self.qed.ntraj)]  # A list of coordinates to update.
        gv = [1.0]*self.qed.ntraj  # A list of values corresponding to the respective
                # coordinate in indices.
        self.qed.set_initial_pure_state(gi, gv)
        #psi = self.qed.initial_state_vec
        #rho = tf.einsum('ijn,kln->ijkln', psi, tf.conj(psi))        
        
        # unitary evolution 
        self.qed.simulation_time = 2.0
        self.qed.dt = 0.01
        expectations_vec, measurements_vec = self.qed.run('vec')
        expectations_mat = self.qed.run('mat_pure')
        self.qed.decoherences['qubit_decay']['measurement_type'] = 'homodyne'
        self.qed.decoherences['qubit_decay']['noise_spectral_density'] = np.ones((1,), dtype=self.qed.rdtype_np)
        expectations_vec_homodyne, measurements_vec_homodyne = self.qed.run('vec')

        from matplotlib.pyplot import plot as plot
        from matplotlib.pyplot import legend as legend
        #print (expectations_vec['qubit_z'])
        plot(np.mean(expectations_vec['qubit_z'], axis=1).T, label='Jumps <n>')
        plot(np.mean(expectations_mat['qubit_z'], axis=1).T, label='ME <n>')
        plot(np.mean(expectations_vec_homodyne['qubit_z'], axis=1).T, label='Homodyne <n>')
        plot(np.mean(expectations_vec['qubit_z_high'], axis=1).T, label='Jumps <n> high-res')
        plot(np.mean(expectations_mat['qubit_z_high'], axis=1).T, label='ME <n> high-res')
        plot(np.mean(expectations_vec_homodyne['qubit_z_high'], axis=1).T, label='Homodyne <n> high-res')
        plot(np.mean(expectations_vec['qubit_x'], axis=1).T, label='Jumps Re{<a>}')
        plot(np.mean(expectations_mat['qubit_x'], axis=1).T, label='ME Re{<a>}')
        plot(np.mean(expectations_vec_homodyne['qubit_x'], axis=1).T, label='Homodyne Re{<a>}')

        legend()

        self.tearDown()
        self.setUp()
        
if __name__ == '__main__':
    unittest.main()