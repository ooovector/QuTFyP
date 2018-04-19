import tensorflow as tf
import tensorflow.contrib.metrics
import numpy
from math import pi
try:
    from tqdm import tqdm
except:
    tqdm = lambda x: x

class TransmonQED:
        # system dynamics in terms of tensorflow functions
    #d = [3, 128] # wavefunction dimensions
    def __init__(self):
        self.coupings = []
        self.measurements = []
        self.rdtype = tf.float32
        self.cdtype = tf.complex64
        self.rdtype_np = numpy.float32
        self.cdtype_np = numpy.complex64
        self.minibatch = 1
    def initialize_operators(self):
        self.d2 = self.d+self.d
        
        # n and sqrt(n) operators (diagonal) on state vectors
        # nice numpy&tensorflow broadcasting
        self.multipliers_sqr_real = [ numpy.reshape([i for i in range(self.d[ax])], 
                [1]*(ax)+[self.d[ax]]+[1]*(len(self.d)-ax))
                for ax in range(len(self.d)) ]
        self.multipliers_real = [numpy.sqrt(m) for m in self.multipliers_sqr_real]
        self.multipliers_sqr = [tf.cast(m, self.cdtype) for m in self.multipliers_sqr_real]
        self.multipliers     = [tf.cast(m, self.cdtype) for m in self.multipliers_real]
        
        # n and sqrt(n) operators on state matrices
        self.multipliers_sqr_real_d2 = [ numpy.reshape([i for i in range(self.d2[ax])], 
                [1]*(ax)+[self.d2[ax]]+[1]*(len(self.d2)-ax))
                for ax in range(len(self.d2)) ]
        self.multipliers_real_d2 = [numpy.sqrt(m) for m in self.multipliers_sqr_real_d2]
        self.multipliers_sqr_d2 = [tf.cast(m, self.cdtype) for m in self.multipliers_sqr_real_d2]
        self.multipliers_d2     = [tf.cast(m, self.cdtype) for m in self.multipliers_real_d2]
    
    ## second quantization operators
    def am(self,x, ax, t='vec'):
        #y = tf.zeros(ax.shape, ax.dtype)
        if t=='vec': 
            d = self.d
            m = self.multipliers
        else: 
            d = self.d2
            m = self.multipliers_d2
        slice_begin = [0]*(len(d)+1)
        #slice2_begin = [0]*len(d)
        slice_size = list(d)+[self.ntraj]
        slice_begin[ax] = 1
        slice_size[ax] = d[ax]-1
        zslice = list(slice_size)
        zslice[ax]=1    
        return tf.concat([tf.slice(x*m[ax], slice_begin, slice_size), tf.zeros(zslice, dtype=self.cdtype)], axis=ax)
    def ap(self,x, ax, t='vec'):
        #y = tf.zeros(ax.shape, ax.dtype)
        if t=='vec': 
            d = self.d
            m = self.multipliers
        else: 
            d = self.d2
            m = self.multipliers_d2
        slice_begin = [0]*(len(d)+1)
        #slice_begin = [0]*len(d)
        slice_size = list(d)+[self.ntraj]
        #slice_begin[ax] = 1
        slice_size[ax] = d[ax]-1
        zslice = list(slice_size)
        zslice[ax]=1
        return tf.concat([tf.zeros(zslice, dtype=self.cdtype), tf.slice(x, slice_begin, slice_size)], axis=ax)*m[ax]
    
    def am_d2(self, x, ax):
        return self.am(x, ax, t='mat')
    def ap_d2(self, x, ax):
        return self.ap(x, ax, t='mat')
    
    def am_td_expect(self, x, t, ax, mode='vec'):
        anh1_shape = [1]*len(x.shape)
        anh1_shape[ax] = self.d[ax]
        transition_frequencies = self.frequencies[ax]+self.anharmonicities[ax]*tf.range(0,self.d[ax], dtype=self.rdtype)
        #state_frequencies = tf.cumsum(transition_frequencies)
        #if mode=='vec':
        anh1m = tf.reshape(transition_frequencies, anh1_shape)
        phase1m = tf.exp(-1j*tf.cast(anh1m, self.cdtype)*tf.cast(t, self.cdtype))
        if mode=='vec':  
            return self.observable(tf.conj(x)*(self.am(x, ax=ax)*phase1m))*2
        else:
            #anh1m = tf.reshape(state_frequencies, anh1_shape)
            #phase1m = tf.exp(-1j*tf.cast(anh1m, self.cdtype)*tf.cast(t, self.cdtype))
            #anh1t_shape = [1]*len(x.shape)
            #anh1t_shape[ax+len(self.d)] = self.d[ax]
            #print (anh1t_shape, anh1_shape)
            #anh1tm = tf.reshape(state_frequencies,   anh1t_shape)
            #phase1tm = tf.exp(1j*tf.cast(anh1tm, self.cdtype)*tf.cast(t, self.cdtype))
            
            #return self.observable(self.am_d2(x, ax=ax)*phase1m*phase1tm, mode='mat')
            return self.observable(self.am_d2(x, ax=ax)*phase1m, mode='mat')*2
        
    
    def Hint(self,x, t):
    # Jaynes-Cummings hamiltonian (part of the Hamiltonian acting upon a ket)
        y = tf.zeros(x.shape, dtype=x.dtype)
        for d1 in range(len(self.d)):
            anh1_shape = [1]*len(x.shape)
            anh1_shape[d1] = self.d[d1]
            anh1p = tf.reshape(self.anharmonicities[d1]*tf.range(-1,self.d[d1]-1, dtype=self.rdtype), anh1_shape)
            anh1m = tf.reshape(self.anharmonicities[d1]*tf.range(0,self.d[d1], dtype=self.rdtype),   anh1_shape)
            phase1p = tf.exp(1j*tf.cast(self.frequencies[d1]+anh1p, self.cdtype)*tf.cast(t, self.cdtype))
            phase1m = tf.exp(-1j*tf.cast(self.frequencies[d1]+anh1m, self.cdtype)*tf.cast(t, self.cdtype))
            ap1 = self.ap(x, ax=d1)*phase1p
            am1 = self.am(x, ax=d1)*phase1m
            for d2 in range(d1+1,len(self.d)):
                anh2_shape = [1]*len(x.shape)
                anh2_shape[d2] = self.d[d2]
                anh2p = tf.reshape(self.anharmonicities[d2]*tf.range(-1,self.d[d2]-1, dtype=self.rdtype), anh2_shape)
                anh2m = tf.reshape(self.anharmonicities[d2]*tf.range(0,self.d[d2], dtype=self.rdtype),   anh2_shape)
                #phase2p = tf.exp(2*pi*1j*tf.cast(self.frequencies[d2]+anh2p, self.cdtype)*tf.cast(t, self.cdtype))
                #phase2m = tf.exp(-2*pi*1j*tf.cast(self.frequencies[d2]+anh2m, self.cdtype)*tf.cast(t, self.cdtype))
                phase2p = tf.exp(1j*tf.cast(self.frequencies[d2]+anh2p, self.cdtype)*tf.cast(t, self.cdtype))
                phase2m = tf.exp(-1j*tf.cast(self.frequencies[d2]+anh2m, self.cdtype)*tf.cast(t, self.cdtype))
                s1 = self.couplings[d1][d2]*phase2p*self.ap(ap1+am1, ax=d2)
                s2 = self.couplings[d1][d2]*phase2m*self.am(ap1+am1, ax=d2)
                #s3 = self.couplings[d1][d2]*phase2p*self.ap(am1, ax=d2)
                #s4 = self.couplings[d1][d2]*phase2m*self.am(am1, ax=d2)
                y += (s1+s2)#+s3+s4)
        return y
    
    def Hint_d2(self, x, t):
        # Jaynes-Cummings hamiltonian (part of the Liouvillian)
        y = tf.zeros(x.shape, dtype=x.dtype)
        for d1 in range(len(self.d)):
            anh1_shape = [1]*len(x.shape)
            anh1_shape[d1] = self.d2[d1]
            anh1t_shape = [1]*len(x.shape)
            anh1t_shape[d1+len(self.d)] = self.d[d1]
            anh1p = tf.reshape(self.anharmonicities[d1]*tf.range(-1,self.d[d1]-1, dtype=self.rdtype), anh1_shape)
            anh1m = tf.reshape(self.anharmonicities[d1]*tf.range(0,self.d[d1], dtype=self.rdtype),   anh1_shape)
            anh1tp = tf.reshape(self.anharmonicities[d1]*tf.range(-1,self.d[d1]-1, dtype=self.rdtype), anh1t_shape)
            anh1tm = tf.reshape(self.anharmonicities[d1]*tf.range(0,self.d[d1], dtype=self.rdtype),   anh1t_shape)
            phase1p = tf.exp(1j*tf.cast(self.frequencies[d1]+anh1p, self.cdtype)*tf.cast(t, self.cdtype))
            phase1m = tf.exp(-1j*tf.cast(self.frequencies[d1]+anh1m, self.cdtype)*tf.cast(t, self.cdtype))
            phase1tp = tf.exp(-1j*tf.cast(self.frequencies[d1]+anh1tp, self.cdtype)*tf.cast(t, self.cdtype))
            phase1tm = tf.exp(1j*tf.cast(self.frequencies[d1]+anh1tm, self.cdtype)*tf.cast(t, self.cdtype))
            # print (phase1p, phase1tp)
            ap1 = self.ap_d2(x, ax=d1)*phase1p
            am1 = self.am_d2(x, ax=d1)*phase1m
            ap1t = self.ap_d2(x, ax=d1+len(self.d))*phase1tp
            am1t = self.am_d2(x, ax=d1+len(self.d))*phase1tm
            for d2 in range(d1+1,len(self.d)):                
                anh2_shape = [1]*len(x.shape)
                anh2_shape[d2] = self.d[d2]
                anh2t_shape = [1]*len(x.shape)
                anh2t_shape[d2+len(self.d)] = self.d[d2]
                anh2p = tf.reshape(self.anharmonicities[d2]*tf.range(-1,self.d[d2]-1, dtype=self.rdtype), anh2_shape)
                anh2m = tf.reshape(self.anharmonicities[d2]*tf.range(0,self.d[d2], dtype=self.rdtype),   anh2_shape)
                anh2tp = tf.reshape(self.anharmonicities[d2]*tf.range(-1,self.d[d2]-1, dtype=self.rdtype), anh2t_shape)
                anh2tm = tf.reshape(self.anharmonicities[d2]*tf.range(0,self.d[d2], dtype=self.rdtype),   anh2t_shape)
                phase2p = tf.exp(1j*tf.cast(self.frequencies[d2]+anh2p, self.cdtype)*tf.cast(t, self.cdtype))
                phase2m = tf.exp(-1j*tf.cast(self.frequencies[d2]+anh2m, self.cdtype)*tf.cast(t, self.cdtype))
                phase2tp = tf.exp(-1j*tf.cast(self.frequencies[d2]+anh2tp, self.cdtype)*tf.cast(t, self.cdtype))
                phase2tm = tf.exp(1j*tf.cast(self.frequencies[d2]+anh2tm, self.cdtype)*tf.cast(t, self.cdtype))
                s3 = self.couplings[d1][d2]*phase2p*self.ap_d2(am1+ap1, ax=d2)
                s4 = self.couplings[d1][d2]*phase2m*self.am_d2(am1+ap1, ax=d2)
                
                #xconj = x
                s7 = self.couplings[d1][d2]*phase2tm*self.am_d2(ap1t+am1t, ax=d2+len(self.d))
                s8 = self.couplings[d1][d2]*phase2tp*self.ap_d2(ap1t+am1t, ax=d2+len(self.d))
                
                y += (s3+s4)-(s7+s8)
        return y

    def V(self,x,t):
        y = tf.zeros(x.shape, dtype=x.dtype)
        # Rabi hamiltonian for external drive (action on ket)
        for cname, c in self.controls.items():
            envelope_value = tf.cast(c['envelope'](t), self.cdtype)
            phase = tf.cast(tf.cos(c['carrier']*t), self.cdtype)
            
            for subsystem_id, couplings in enumerate(c['coupling']):
                anh_shape = [1]*len(x.shape)
                anh_shape[subsystem_id] = self.d[subsystem_id]
                anhp = tf.reshape(self.anharmonicities[subsystem_id]*tf.range(-1,self.d[subsystem_id]-1, dtype=self.rdtype), anh_shape)
                anhm = tf.reshape(self.anharmonicities[subsystem_id]*tf.range(0,self.d[subsystem_id], dtype=self.rdtype),   anh_shape)
                phasep = tf.exp(1j*tf.cast(self.frequencies[subsystem_id]+anhp, self.cdtype)*tf.cast(t, self.cdtype))
                phasem = tf.exp(-1j*tf.cast(self.frequencies[subsystem_id]+anhm, self.cdtype)*tf.cast(t, self.cdtype))
                #phase_subsystem_p = tf.exp(2*pi*1j*tf.cast(frequencies[subsystem_id], cdtype)*tf.cast(t, cdtype))
                for coupling_type, coupling in couplings.items():
                    coupling_cmplx = tf.cast(coupling, self.cdtype)
                    constant = envelope_value*phase*coupling_cmplx

                    if coupling_type == 'x':
                        action = self.am(x,ax=subsystem_id)*phasem+self.ap(x,ax=subsystem_id)*phasep
                    elif coupling_type == 'y':
                        action = 1j*(self.am(x,ax=subsystem_id)*phasem-self.ap(x,ax=subsystem_id)*phasep)
                    elif coupling_type == 'z':
                        action = x*self.multipliers_sqr[subsystem_id]
                    y += action*constant
        return y
    
    def V_d2(self, x,t):
        y = tf.zeros(x.shape, dtype=x.dtype)
        # Rabi hamiltonian for external drive (action on state matrix in Liouvillian)
        for cname, c in self.controls.items():
            envelope_value = c['envelope'](t)
            for subsystem_id, couplings in enumerate(c['coupling']):
                #phase_subsystem = tf.exp(2*pi*1j*tf.cast(self.frequencies[subsystem_id], self.cdtype)*tf.cast(t, self.cdtype))
                anh_shape = [1]*len(x.shape)
                anh_shape[subsystem_id] = self.d[subsystem_id]
                anht_shape = [1]*len(x.shape)
                anht_shape[subsystem_id+len(self.d)] = self.d[subsystem_id]
                anhp = tf.reshape(self.anharmonicities[subsystem_id]*tf.range(-1,self.d[subsystem_id]-1, dtype=self.rdtype), anh_shape)
                anhm = tf.reshape(self.anharmonicities[subsystem_id]*tf.range(0,self.d[subsystem_id], dtype=self.rdtype),   anh_shape)
                anhtp = tf.reshape(self.anharmonicities[subsystem_id]*tf.range(-1,self.d[subsystem_id]-1, dtype=self.rdtype), anht_shape)
                anhtm = tf.reshape(self.anharmonicities[subsystem_id]*tf.range(0,self.d[subsystem_id], dtype=self.rdtype),   anht_shape)
                phasep = tf.exp(1j*tf.cast(self.frequencies[subsystem_id]+anhp, self.cdtype)*tf.cast(t, self.cdtype))
                phasem = tf.exp(-1j*tf.cast(self.frequencies[subsystem_id]+anhm, self.cdtype)*tf.cast(t, self.cdtype))
                phasetp = tf.exp(-1j*tf.cast(self.frequencies[subsystem_id]+anhtp, self.cdtype)*tf.cast(t, self.cdtype))
                phasetm = tf.exp(1j*tf.cast(self.frequencies[subsystem_id]+anhtm, self.cdtype)*tf.cast(t, self.cdtype))
                
                for coupling_type, coupling in couplings.items():
                    if numpy.any(numpy.abs(coupling) > 0.0):
                        phase = tf.cos(tf.cast(c['carrier'], self.cdtype)*tf.cast(t, self.cdtype))
                        constant = tf.cast(envelope_value, self.cdtype)*phase*tf.cast(coupling, self.cdtype)
                        amx = self.am_d2(x,ax=subsystem_id)*phasem
                        apx = self.ap_d2(x,ax=subsystem_id)*phasep
                        apxt = self.ap_d2(x,ax=subsystem_id+len(self.d))*phasetp
                        amxt = self.am_d2(x,ax=subsystem_id+len(self.d))*phasetm
                        if coupling_type == 'x':
                            action_p = amx+apx
                            action_m = apxt+amxt
                        elif coupling_type == 'y':
                            action_p = 1j*(amx-apx)
                            action_m = 1j*(apxt-amxt)
                        elif coupling_type == 'z':
                            action_p = x*self.multipliers_sqr[subsystem_id]
                            action_m = x*self.multipliers_sqr[subsystem_id+len(self.d)]
                        y += (action_p-action_m)*constant
        return y
        
    def Hnonh(self,x,t):
        # nonhermitian effective hamilotnian due to decay and homodyne detection
        y = tf.zeros(x.shape, dtype=x.dtype)
        #for decoherence_id, decoherence in enumerate(self.decoherences):
        for decoherence_id, decoherence in self.decoherences.items():
            # 0: T1 (decay) decoherence via am
            if decoherence['coupling_type'] == 'a':
                rate = decoherence['rate']
                subsystem_id = decoherence['subsystem_id']
                y += 0.5*rate*self.multipliers_sqr[subsystem_id]*x#ap(am(x, ax=subsystem_id))
        return -1j*y
    
    def L(self, x,t):
        y = tf.zeros(x.shape, dtype=x.dtype)
        #for decoherence_id, decoherence in enumerate(self.decoherences):
        for decoherence_id, decoherence in self.decoherences.items():
            if decoherence['coupling_type'] == 'a':
                rate = decoherence['rate']
                subsystem_id = decoherence['subsystem_id']
                y += rate*(self.am_d2(self.am_d2(x,ax=subsystem_id),ax=subsystem_id+len(self.d))-\
                     0.5*(self.multipliers_sqr_d2[subsystem_id]+self.multipliers_sqr_d2[subsystem_id+len(self.d)])*x)
        return 1j*y
    
    def calc_norm(self,x, mode='vec'):
        if mode=='vec':
            return tf.sqrt(tf.reduce_sum(tf.real(x)**2+tf.imag(x)**2, axis=[i for i in range(len(self.d))]))
        else:
            return tf.trace(tf.transpose(tf.reshape(tf.real(x), [numpy.prod(self.d), numpy.prod(self.d), self.ntraj]), [2, 0, 1]))
        
    def dof_ravel(self, x, mode='vec'):
        if mode == 'vec':
            return tf.reshape(x, (numpy.prod(self.d), self.ntraj))
        else:
            return tf.reshape(x, (numpy.prod(self.d), numpy.prod(self.d), self.ntraj))
    
    def observable(self, x, mode='vec'):
        if mode=='vec': 
            o = tf.reduce_sum(self.dof_ravel(x), axis=0)
        else:
            o = tf.trace(tf.transpose(self.dof_ravel(x, mode='mat'), [2, 0, 1]))
        return o
    
    def calc_expectations(self, x, t, mode='vec'):
        expectations = {}
        for name, exp in self.expectations.items():
            if mode=='vec':
                expectations[name] = exp['observable_vec'](x,t)
            else:
                expectations[name] = exp['observable_mat'](x,t)
        return expectations
    
    # old expectation code: measure all single-particle observables, no downsampling, result in tf tensor
    def old_expect_sp(self,x, mode='vec'):
        if mode=='vec':
            expectations = []
            for particle_id in range(len(self.d)):
                am_expect = tf.reduce_sum(self.dof_ravel(tf.conj(x)*self.am(x, ax=particle_id)), axis=0)
                n_expect = tf.reduce_sum(self.dof_ravel((tf.real(x)**2+tf.imag(x)**2)*self.multipliers_sqr_real[particle_id]), axis=0)
                #am_expect = (tf.reduce_sum(tf.conj(x)*self.am(x, ax=particle_id), axis=[i for i in range(len(self.d))]))
                particle = [tf.real(am_expect),
                            tf.imag(am_expect),
                            n_expect]
                expectations.extend(particle)
            return expectations
        else:
            expectations = []
            for particle_id in range(len(self.d)):
                am_expect = tf.trace(tf.transpose(self.dof_ravel(self.am_d2(x, ax=particle_id), mode='mat'), [2, 0, 1]))
                n_expect = tf.trace(tf.transpose(self.dof_ravel(tf.real(x)*self.multipliers_sqr_real[particle_id], mode='mat'), [2, 0, 1]))
                
                #am_act = tf.transpose(tf.reshape(self.am_d2(x, ax=particle_id), [numpy.prod(self.d), numpy.prod(self.d), self.ntraj]), [2, 0, 1])
                #am_expect = tf.trace(am_act)
                #ap_act = tf.transpose(tf.reshape(ap(x, ax=particle_id), [np.prod(d), np.prod(d), ntraj]), [2, 0, 1])
                #n_act = tf.transpose(tf.reshape(tf.real(x)*self.multipliers_sqr_real[particle_id], [numpy.prod(self.d), numpy.prod(self.d), self.ntraj]), [2, 0, 1])
                particle = [tf.real(am_expect), tf.imag(am_expect), n_expect]
                expectations.extend(particle)
            return expectations
    
    def H(self, x,t,quantum_noise):
        H1 = self.Hint(x,t)+self.V(x,t)+self.Hnonh(x,t)
        H2,J = self.homodyne_conditional(x, t, quantum_noise)
        return H1+H2, J
    
    def H_d2(self, x, t):
        return self.Hint_d2(x,t)+self.V_d2(x,t)+self.L(x,t)
    
    def jump(self, x, expectations):
        #for decoherence_id, decoherence in enumerate(self.decoherences):
        J = {}
        for decoherence_id, decoherence in self.decoherences.items():
            if decoherence['measurement_type'] == 'photon-counting':
                if decoherence['coupling_type'] == 'a':
                    rate = decoherence['rate']
                    subsystem_id = decoherence['subsystem_id']
                    #normalized_vector = x/tf.sqrt(x);
                    expectation = tf.reduce_sum((tf.real(x)**2+tf.imag(x)**2)*
                             tf.cast(self.multipliers_sqr[subsystem_id], self.rdtype), axis=[i for i in range(len(self.d))])
                    probability = expectation*(rate*self.dt)
                    all_probabilities =  tf.transpose(tf.concat([[1-probability, probability]], axis=0))
                                                                                             
                    jump = tf.cast(tf.transpose(tf.multinomial(tf.log(all_probabilities), 1)), dtype=self.cdtype)
                    x = self.am(x, ax=subsystem_id)*jump+x*(1-jump)
                    J[decoherence_id] = tf.cast(jump, dtype=self.rdtype)
        return x, J

    def homodyne_conditional(self, x, t, quantum_noise):
        y = tf.zeros(x.shape, dtype=self.cdtype)
        homodyne_decoherence_id = 0;
        #for decoherence_id, decoherence in enumerate(self.decoherences):
        J = {}
        for decoherence_id, decoherence in self.decoherences.items():
            if decoherence['measurement_type'] == 'homodyne':
                print ('homodyne')
                if decoherence['coupling_type'] == 'a':
                    rate = decoherence['rate']
                    subsystem_id = decoherence['subsystem_id']
                    anh_shape = [1]*len(x.shape)
                    anh_shape[subsystem_id] = self.d[subsystem_id]
                    anhm = tf.reshape(self.anharmonicities[subsystem_id]*tf.range(0,self.d[subsystem_id], dtype=self.rdtype),   anh_shape)
                    phasem = tf.exp(-1j*tf.cast(self.frequencies[subsystem_id]+anhm, self.cdtype)*tf.cast(t, self.cdtype))
                    op  = numpy.sqrt(rate)*self.am(x, ax=subsystem_id)*phasem
                    expect = 2*tf.real(tf.reduce_sum(tf.conj(x)*op, axis=[i for i in range(len(self.d))]))
                    measurement = (tf.reshape(quantum_noise[:,homodyne_decoherence_id], (-1,))/numpy.sqrt(self.dt) + expect) # measurement_record
                    J[decoherence_id] = measurement
                    homodyne_decoherence_id += 1;
                else:
                    raise ValueError('Coupling type {} not supported for homodyne detection.'.format(decoherence['coupling_type']))
                y += op*tf.cast(measurement, self.cdtype)
        return 1j*y, J
                    
    # perform one iteration of RK4
    def odeint_iteration(self, x, t, quantum_noise, mode='vec'):
        dt = self.dt
        if mode=='vec':
            H = self.H
            args = (quantum_noise,)
            H1, J1 = H(x, t, *args)
            f1 = -1j*H1*self.dt
            H2, J2 = H(x+f1/2., t+self.dt/2., *args)
            f2 = -1j*H2*dt
            H3, J3 = H(x+f2/2., t+self.dt/2., *args)
            f3 = -1j*H3*dt
            H4, J4 = H(x+f3, t+self.dt, *args)
            f4 = -1j*H4*dt
        elif mode=='mat_pure' or mode=='mat':
            H = self.H_d2
            args = tuple([])
            f1 = -1j*H(x, t, *args)*self.dt
            f2 = -1j*H(x+f1/2., t+self.dt/2., *args)*dt
            f3 = -1j*H(x+f2/2., t+self.dt/2., *args)*dt
            f4 = -1j*H(x+f3, t+self.dt, *args)*dt

        x_new = x+(f1+f2*2+f3*2+f4)/6.
        if mode=='vec':
            J_average = { k:(J1[k]+2*J2[k]+2*J3[k]+J4[k])/6. for k in J1.keys() }
        
        norm = self.calc_norm(x_new, mode=mode)
        x = tf.assign(x, x_new/tf.cast(norm, self.cdtype))
        
        # old expectation code
        #expectations = self.old_expect_sp(x, mode=mode)
        expectations = self.calc_expectations(x, t,  mode=mode)

        # check if random jump occurs
        # uncomment if homodyne detection is off
        if mode=='vec':
            x_jumped, J  = self.jump (x, expectations)
            x= tf.assign(x, x_jumped)
            J_average.update(J)

        #return x, tf.stack(expectations), J_average
        
        self.expectation_fetch_order = [e for e in expectations.keys()]
        expectations = [expectations[e] for e in self.expectation_fetch_order]
        results = expectations
        if mode=='vec':
            self.measurement_fetch_order = [m for m in J_average.keys()]
            measurements = [J_average[m] for m in self.measurement_fetch_order]
            results += measurements
         
        return tuple(results) # workaraound for tensorflow that doesn't return a dict
        # but rather a tuple
    
    def pure_state(self, gi, gv):
        return tf.Variable(tf.cast(tf.sparse_tensor_to_dense(tf.SparseTensor(gi, gv, self.d+[self.ntraj])), self.cdtype))
       
    
    def set_initial_pure_state(self, gi, gv, mat=True):
        with tf.Session() as sess:
            x = tf.Variable(tf.cast(tf.sparse_tensor_to_dense(tf.SparseTensor(gi, gv, self.d+[self.ntraj])), self.cdtype), name='initial_pure_state_tensor')
            sess.run(x.initializer)
            self.initial_state_vec = tf.Variable(sess.run(x), name='initial_pure_state_vec_variable')
            sess.run(self.initial_state_vec.initializer)
            if mat:
                x_flat1 = tf.reshape(x, [numpy.prod(self.d), 1, self.ntraj])
                x_flat2 = tf.reshape(x, [1, numpy.prod(self.d), self.ntraj])
                self.initial_state_mat = tf.Variable(sess.run(tf.reshape(x_flat1*tf.conj(x_flat2), self.d+self.d+[self.ntraj])), name='initial_pure_state_mat_variable')
            #sess.run(self.initial_state_mat.initializer)
        
    #def set_simulation_parameters(self):
    @property
    def opts(self):
        o = {'simulation_time': self.simulation_time,
             'dt': self.dt,
             'ntraj': self.ntraj,
             'rdtype': self.rdtype,
             'cdtype': self.cdtype}
        return o
    
    @opts.setter
    def opts(self, o):
        self.simulation_time = o['simulation_time']
        self.dt = o['dt']
        self.ntraj = o['ntraj']
        if 'rdtype' in o:
            self.rdtype = o['rdtype']
        if 'cdtype' in o:
            self.cdtype = o['cdtype']
    
    def downsampled_minibatch_shape(self, measurement):
        if measurement['unmix']:
            #freqs = numpy.fft.fftfreq(self.minibatch, self.dt)
            #print('fftfreqs: ', freqs)
            #nonzero_freqs = numpy.logical_or(numpy.logical_and(freqs>=measurement['window_start'], freqs<=measurement['window_stop']),
            #                                 numpy.logical_and(freqs>=-measurement['window_start'], freqs>=-measurement['window_stop']))
            #print (numpy.sum(nonzero_freqs), len(nonzero_freqs))
            #return numpy.sum(nonzero_freqs)
            
            # current sample rate:
            old_sample_period = self.dt
            new_sample_period = 1/measurement['sample_rate']
            
            average_size = int(new_sample_period/old_sample_period)
            
            return int(self.minibatch/average_size)
        return self.minibatch
        
    def downsample(self, measurement, rf, t):
        if measurement['unmix']:
            time_rf = numpy.reshape(numpy.arange(self.minibatch)*self.dt+t, (self.minibatch, 1))
            prod = (rf * numpy.exp(1j*measurement['unmix_reference']*time_rf)).T
            #print (prod.shape)
            iq_cmplx = numpy.mean(numpy.reshape(prod, (self.ntraj, int(self.minibatch*(self.dt*measurement['sample_rate'])), 
                                                       int(1/(self.dt*measurement['sample_rate'])))), axis=2).T
            
            #freqs = numpy.fft.fftfreq(self.minibatch, self.dt)
            #nonzero_freqs = numpy.logical_or(numpy.logical_and(freqs>=measurement['window_start'], freqs<=measurement['window_stop']),
            #                                 numpy.logical_and(freqs>=-measurement['window_start'], freqs>=-measurement['window_stop']))
            #downsampled = numpy.fft.ifft(numpy.fft.fft(upsampled, axis=0, norm=None)[nonzero_freqs,:]/numpy.sqrt(self.minibatch), axis=0, norm=None)*numpy.sqrt(sum(nonzero_freqs))
            #print (downsampled.shape, upsampled.shape)
            #return downsampled
            return iq_cmplx
        return rf
    
    def run(self, mode='vec', interactive=False):
        sess = tf.Session()
        
        tsteps = numpy.arange(0, self.simulation_time, self.dt, dtype=self.rdtype_np)
        #expectations = numpy.zeros((len(self.d)*3, len(tsteps), self.ntraj), dtype=self.rdtype_np)
        expectations = {expectation_name: numpy.zeros((int(self.downsampled_minibatch_shape(self.expectations[expectation_name])*len(tsteps)/self.minibatch), self.ntraj), 
                                dtype=(self.cdtype_np if self.expectations[expectation_name]['unmix'] else self.rdtype_np)) 
                                for expectation_name in self.expectations }
        expectations_minibatch = {expectation_name:  numpy.zeros((self.minibatch, self.ntraj), dtype=self.rdtype_np) 
                        for expectation_name in self.expectations }
        
        if mode != 'mat':
            measurements = {measurement_name: numpy.zeros((int(self.downsampled_minibatch_shape(self.decoherences[measurement_name])*len(tsteps)/self.minibatch), self.ntraj), 
                                dtype=(self.cdtype_np if self.decoherences[measurement_name]['unmix'] else self.rdtype_np)) 
                                for measurement_name in self.decoherences.keys() }
            measurements_minibatch = {measurement_name: numpy.zeros((self.minibatch, self.ntraj), dtype=self.rdtype_np) 
                            for measurement_name in self.decoherences }
        
        #print({k:m.shape for k,m in measurements.items()})
        # all randoms are related to measurements
        #randoms = numpy.zeros((len(tsteps), ntraj, len(d), 3), dtype=numpy.float32)
        nmeas = len(self.decoherences)
        #measurements = numpy.zeros((nmeas, len(tsteps), self.ntraj), dtype=self.rdtype)
        if mode=='vec':
            x = self.initial_state_vec
            initial = x
        elif mode=='mat_pure':
            x = self.initial_state_mat
            initial = self.initial_state_mat
        else:
            raise(ValueError('Initial state type unrecognized'))
        t = tf.placeholder(dtype=tf.float32, shape=[])
        sess.run(initial.initializer)
#        fig = plt.figure('Average photon count in qubit')
#        line, = plt.plot(drive_frequencies, np.mean(expectations[2,:,:], axis=0))
#        plt.autoscale(True)
#        plt.show()

        gaussian_environments = len([d for d in self.decoherences.values() if d['measurement_type'] == 'homodyne'])
        #quantum_noise = tf.random_normal([self.ntraj, gaussian_environments], 
        #                                 mean=0.0, 
        #                                 stddev=1.0, 
        #                                 dtype=self.rdtype, 
        #                                 seed=None, 
        #                                 name='quantum_noise')
        white_noise = tf.complex(tf.random_normal([self.ntraj, gaussian_environments, self.minibatch], 
                                         mean=0.0, 
                                         stddev=1.0, 
                                         dtype=self.rdtype, 
                                         seed=None, 
                                         name='quantum_noise_spectral_real'),
                                    tf.random_normal([self.ntraj, gaussian_environments, self.minibatch], 
                                         mean=0.0, 
                                         stddev=1.0, 
                                         dtype=self.rdtype, 
                                         seed=None, 
                                         name='quantum_noise_spectral_imag'))
        
        noise = tf.real(tf.ifft(tf.fft(white_noise)), name='correlated_noise_time_domain')
        #noise = tf.random_normal([self.ntraj, gaussian_environments, self.minibatch], 
        #                                 mean=0.0, 
        #                                 stddev=1.0, 
        #                                 dtype=self.rdtype, 
        #                                 seed=None, 
        #                                 name='quantum_noise_spectral_imag')
        noise_instant = tf.placeholder(shape=[self.ntraj, gaussian_environments], dtype=self.rdtype)
        #mini_id = tf.placeholder(dtype=tf.int32, shape = [])
        
        iteration = self.odeint_iteration(x, t, noise_instant, mode=mode)
        
        for t_id, tit in enumerate(tqdm(tsteps)):
            # check if we are entering new noise/resample minibatch
            if not (t_id % self.minibatch):
                #regenerate noise
                noise_realization = sess.run(noise)
                
            step = sess.run(iteration, feed_dict={t:tit, noise_instant:noise_realization[:,:,t_id%self.minibatch]})
            #, jumps_new, jump_proba, masks_new, x_jumped = sess.run(iteration, feed_dict={t:tit})
            #psi_new = step[0]
            
            for expectation_id, expectation in enumerate(self.expectation_fetch_order):
                #print (step[expectation_id+1])
                #print (expectations_minibatch[expectation][t_id % self.minibatch])
                expectations_minibatch[expectation][t_id % self.minibatch] = step[expectation_id]
            if mode != 'mat' and mode != 'mat_pure':
                for measurement_id, measurement in enumerate(self.measurement_fetch_order):
                    measurements_minibatch[measurement][t_id % self.minibatch] = \
                        step[measurement_id+len(self.expectation_fetch_order)]
            # old expectation code
            #expectations[:,t_id,:] = expectations_new
            # new expectation code
            #print (self.measurement_fetch_order, measurements_minibatch)
            
            if not ((t_id+1) % self.minibatch):
                # downsample measurements and expectatons
                if mode != 'mat' and mode != 'mat_pure':
                    for measurement_name, measurement in self.decoherences.items():
                        downsampled_size = self.downsampled_minibatch_shape(measurement)
                        #print (measurements[measurement_name].shape, (t_id//self.minibatch)*downsampled_size, \
                        #           (t_id//self.minibatch+1)*downsampled_size)
                        measurements[measurement_name][(t_id//self.minibatch)*downsampled_size:\
                                                       ((t_id//self.minibatch)+1)*downsampled_size,:] = \
                            self.downsample(measurement, measurements_minibatch[measurement_name], tit-self.dt*self.minibatch)
                        pass
                for expectation_name, expectation in self.expectations.items():
                    downsampled_size = self.downsampled_minibatch_shape(expectation)
                    expectations[expectation_name][(t_id//self.minibatch)*downsampled_size:\
                                                       ((t_id//self.minibatch)+1)*downsampled_size,:] = \
                        self.downsample(expectation, expectations_minibatch[expectation_name], tit-self.dt*self.minibatch)
                    pass
            #if mode != 'mat' and mode != 'mat_pure':
            #    print (numpy.sum(numpy.sum(numpy.abs(psi_new)**2, axis=0), axis=1))

#            masks.append(masks_new)
#            if t_id%500==0:
#                line.set_ydata(np.mean(expectations[2,:t_id,:], axis=0))
#                fig.canvas.draw()
#                plt.gca().relim()
#                plt.pause(0.01)
            #print (psi_new)
        sess.close()
        if mode != 'mat' and mode != 'mat_pure':
            return  expectations, measurements#{k:v for k,v in zip(self.expectation_fetch_order, expectations)}, \
                    #{k:v for k,v in zip(self.measurement_fetch_order, measurements)}
        else:
            return  expectations#{k:v for k,v in zip(self.expectation_fetch_order, expectations)}
