# SummerInternship2023

# Quantum Error Correction
Quantum error correction comes from the marriage of quantum mechanics with the classical theory of error correcting codes. Error correction which is a central part of classical information theory is similarly a foundation of quantum information theory.
It is concerned with the fundamental problem of communication and information storage. Error correction is of an important use in quantum computers because the algorithms make use of large-scale quantum interference, which is fragile.
The first quantum error correcting codes were discovered independently by Shor and steane. Shor proved that 9 qubits can be used to protect a single qubit from errors while steane code does the same with only 7 qubits.

## Three Bit code
Here we'll introduce the concept of Noise Model module which is an inbuilt module in Qiskit in which it can create noise in any circuit.

```python
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error

def get_noise(p_meas,p_gate):

    error_meas = pauli_error([('X',p_meas), ('I', 1 - p_meas)])
    error_gate1 = depolarizing_error(p_gate, 1)
    error_gate2 = error_gate1.tensor(error_gate1)

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_meas, "measure") # measurement error is applied
    noise_model.add_all_qubit_quantum_error(error_gate1, ["x"]) # single qubit gate error is applied to x gates
    noise_model.add_all_qubit_quantum_error(error_gate2, ["cx"]) # two qubit gate error is applied to cx gates
        
    return noise_model
noise_model = get_noise(0.05,0.05)
```
An implementation of noise error i.e., storing one qubit into 3 qubits.
```python
from qiskit import QuantumCircuit, execute, Aer

qc0 = QuantumCircuit(3,3,name='0') # initialization of circuit

qc0.measure(qc0.qregs[0],qc0.cregs[0]) # measure the qubits

# run the circuit and extract the noise counts
counts = execute( qc0, Aer.get_backend('qasm_simulator'),noise_model=noise_model).result().get_counts()

print(counts)
plot_histogram(counts)
```
The code to solve this bit flip issue is
```python
from qiskit import QuantumRegister, ClassicalRegister

cq = QuantumRegister(2, 'code_qubit')
lq = QuantumRegister(1, 'ancilla_qubit')
sb = ClassicalRegister(1, 'syndrome_bit') #syndrome nits are just use for measurement
qc = QuantumCircuit(cq, lq, sb)
qc.cx(cq[0], lq[0])
qc.cx(cq[1], lq[0])
qc.measure(lq, sb)
qc.draw()
```
Now we'll use ibm real simulator to get our results. we'll mae use of ibm_lagos
```python
# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, transpile
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from ibm_quantum_widgets import *
from typing import List, Optional

from qiskit import transpile, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.result import marginal_counts

import warnings

warnings.filterwarnings("ignore")
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Estimator, Session, Options

# Loading your IBM Quantum account(s)
service = QiskitRuntimeService(channel="ibm_quantum")

from qiskit_ibm_provider import IBMProvider
provider = IBMProvider()
hub = "ibm-q"
group = "open"
project = "main"
hgp = f"{hub}/{group}/{project}"
backend_name = "ibm_lagos"

backend = provider.get_backend(backend_name, instance=hgp)
print(f"Using {backend.name}")
```
we'll setup a base quantum circuit for our experiments
```python
qreg_data = QuantumRegister(3)
qreg_measure = QuantumRegister(2)
creg_data = ClassicalRegister(3)
creg_syndrome = ClassicalRegister(2)
state_data = qreg_data[0]
ancillas_data = qreg_data[1:]


def build_qc() -> QuantumCircuit:
    return QuantumCircuit(qreg_data, qreg_measure, creg_data, creg_syndrome)
```
```python
#initializing process
qc_init = build_qc()

qc_init.x(qreg_data[0])
qc_init.barrier(qreg_data)

qc_init.draw(output="mpl", idle_wires=False)
```
```python
#encoding of our logical state to protect it
#the circuit below maos it into ather state
def encode_bit_flip(qc, state, ancillas):
    control = state
    for ancilla in ancillas:
        qc.cx(control, ancilla)
    qc.barrier(state, *ancillas)
    return qc


qc_encode_bit = build_qc()

encode_bit_flip(qc_encode_bit, state_data, ancillas_data)

qc_encode_bit.draw(output="mpl", idle_wires=False)
```
```python
def decode_bit_flip(qc, state, ancillas):
    inv = qc_encode_bit.inverse()
    return qc.compose(inv)


qc_decode_bit = build_qc()

qc_decode_bit = decode_bit_flip(qc_decode_bit, state_data, ancillas_data)

qc_decode_bit.draw(output="mpl", idle_wires=False)
```
```python
def measure_syndrome_bit(qc, qreg_data, qreg_measure, creg_measure):
    qc.cx(qreg_data[0], qreg_measure[0])
    qc.cx(qreg_data[1], qreg_measure[0])
    qc.cx(qreg_data[0], qreg_measure[1])
    qc.cx(qreg_data[2], qreg_measure[1])
    qc.barrier(*qreg_data, *qreg_measure)
    qc.measure(qreg_measure, creg_measure)
    qc.x(qreg_measure[0]).c_if(creg_measure[0], 1)
    qc.x(qreg_measure[1]).c_if(creg_measure[1], 1)
    qc.barrier(*qreg_data, *qreg_measure)
    return qc


qc_syndrome_bit = measure_syndrome_bit(
    build_qc(), qreg_data, qreg_measure, creg_syndrome
)
qc_syndrome_bit.draw(output="mpl", idle_wires=False)
```
```python
qc_measure_syndrome_bit = qc_encoded_state_bit.compose(qc_syndrome_bit)
qc_measure_syndrome_bit.draw(output="mpl", idle_wires=False)
```
```python
#correction application
def apply_correction_bit(qc, qreg_data, creg_syndrome):
    qc.x(qreg_data[0]).c_if(creg_syndrome, 3)
    qc.x(qreg_data[1]).c_if(creg_syndrome, 1)
    qc.x(qreg_data[2]).c_if(creg_syndrome, 2)
    qc.barrier(qreg_data)
    return qc


qc_correction_bit = apply_correction_bit(build_qc(), qreg_data, creg_syndrome)
qc_correction_bit.draw(output="mpl", idle_wires=False)
```
```python
def apply_final_readout(qc, qreg_data, creg_data):
    qc.barrier(qreg_data)
    qc.measure(qreg_data, creg_data)
    return qc


qc_final_measure = apply_final_readout(build_qc(), qreg_data, creg_data)
qc_final_measure.draw(output="mpl", idle_wires=False)
```
```python
bit_code_circuit = qc_measure_syndrome_bit.compose(qc_correction_bit).compose(
    qc_final_measure
)
bit_code_circuit.draw(output="mpl", idle_wires=False)
```
This will give us the final circuit of the three bit code required.
## Shor's Code
The Shor code is a 9-qubit circuit that requires 8 ancillary qubits to correct 1 qubit. For simplification we will call the 1st qubit that we want to correct the main qubit and the ancillary qubits 1 to 8. It's a combination of 6 bits where 3 bits are used for bit flip error correction and other 3 are used for phase flip error correction.

The general overview circuitof shor code is given below.
```python
print('\nShor Code')
print('--------------')

from qiskit import QuantumRegister
from qiskit import ClassicalRegister
from qiskit import QuantumCircuit, execute,IBMQ
from qiskit.tools.monitor import job_monitor

IBMQ.enable_account('e1528aa940d0e33053ad844def737d79a6c054cb6e5eabdeaebcd905efff73e432748f992e5db2a270a8f6ffa8281f88f66c26d7fbfaf1f1fe874a1700bb8c6c')
provider = IBMQ.get_provider(hub='ibm-q')

backend = provider.get_backend('ibmq_qasm_simulator')

q = QuantumRegister(1,'q') #this is just a one qubit circuit
c = ClassicalRegister(1,'c')

circuit = QuantumCircuit(q,c)

circuit.h(q[0]) #to create superposition

#error here
circuit.x(q[0]) #Bit flip error
circuit.z(q[0]) #Phase flip error

circuit.h(q[0]) #to get back into original state

circuit.barrier(q)

circuit.measure(q[0],c[0])

job = execute(circuit, backend, shots=1000)

job_monitor(job)

counts = job.result().get_counts()

print("\n Uncorrected bit flip and phase error")
print("--------------------------------------")
print(counts)

#shor code starts
q = QuantumRegister(9,'q') #9 qubit used in which 8 of them are ancillary
c = ClassicalRegister(1,'c')

circuit = QuantumCircuit(q,c)

circuit.cx(q[0],q[3]) #two CNOT gates
circuit.cx(q[0],q[6])

circuit.h(q[0]) #to create superposition
circuit.h(q[3])
circuit.h(q[6])

circuit.cx(q[0],q[1])
circuit.cx(q[3],q[4])
circuit.cx(q[6],q[7])

circuit.cx(q[0],q[2])
circuit.cx(q[3],q[5])
circuit.cx(q[6],q[8])

circuit.barrier(q)

#error here
circuit.x(q[0]) #Bit flip error
circuit.z(q[0]) #Phase flip error


circuit.barrier(q)
circuit.cx(q[0],q[1])
circuit.cx(q[3],q[4])
circuit.cx(q[6],q[7])

circuit.cx(q[0],q[2])
circuit.cx(q[3],q[5])
circuit.cx(q[6],q[8])

circuit.ccx(q[1],q[2],q[0]) #toffoli gates
circuit.ccx(q[4],q[5],q[3])
circuit.ccx(q[8],q[7],q[6])

circuit.h(q[0]) #to return back into original states
circuit.h(q[3])
circuit.h(q[6])

circuit.cx(q[0],q[3])
circuit.cx(q[0],q[6])
circuit.ccx(q[6],q[3],q[0])

circuit.barrier(q)

circuit.measure(q[0],c[0]) #input is given

circuit.draw(output='mpl',filename='shorcode.png') #Draws an image of the circuit

job = execute(circuit, backend, shots=1000)

job_monitor(job)

counts = job.result().get_counts()


print("\nShor code with bit flip and phase error")
print("----------------------------------------")
print(counts)
input()
circuit.draw()
```

## Steane Code
It can protect a qubit against a bit-flip and a phase-flip using 6 spare qubits. It is also defined as the [7,1,3] Hamming code as it store one qubit into 7 qubits with distance as 3.
The codewords are 

    |0> = |0000000> +|1010101> + |0110011> + |1100110> + 
          |0001111> + |1011010> + |0111100> + |1101001>
    |1> = 𝑥_1111111 |0>
If there are errors in it, then we can see change in codewords. 
Now, we can see as we have 1 main qubit and 6 ancillary qubit so the no. of logical qubit it can keep safe is 1.

Encoding Circuit
```python
cq = QuantumRegister(1, 'A')
lq = QuantumRegister(6, 'B')
#get it into superposition states
qc.h(lq[3])
qc.h(lq[4])
qc.h(lq[5])

qc=MCMT('cx',cq, 2[lq[0], lq[1]])
qc=MCMT('cx',lq[5], 3[lq[0], lq[2], cq])
qc=cccx('cx',lq[4], 3[cq, lq[1], lq[2]])
qc=('cx', lq[3], 3[lq[0], lq[1], lq[2]])

qc.draw()
```
Error Correcting Circuit
```python
cq= QuantumRegister(7, 'A')
lq= QuantumRegister(6, 'B')

#Bit flip detection
#BIT2
qc.cx(cq[0],lq[0])
qc.cx(cq[2],lq[0])
qc.cx(cq[4],lq[0])
qc.cx(cq[6],lq[0])
#BIT1
qc.cx(cq[1],lq[0])
qc.cx(cq[2],lq[0])
qc.cx(cq[5],lq[0])
qc.cx(cq[6],lq[0])
#BIT0
qc.cx(cq[3],lq[0])
qc.cx(cq[4],lq[0])
qc.cx(cq[5],lq[0])
qc.cx(cq[6],lq[0])

#Phase flip error correction
#into superposition states
qc.h(lq[3])
qc.h(lq[4])
qc.h(lq[5])

qc=MCMT('cx',lq[3], 4[cq[0], cq[2],cq[4],cq[6]])
qc=MCMT('cx',lq[4], 4[cq[1], cq[2],cq[5],cq[6]])
qc=cccx('cx',lq[5], 4[cq[3], cq[4],cq[5],cq[6]])
#maps back from superposition states
qc.h(lq[3])
qc.h(lq[4])
qc.h(lq[5])

qc.draw()
```

## Surface Code
The challenge in creating QECC lies in finding commuting sets of stabilizers that enable errors to be detected without disturbing the encoded information.
Surface code belongs to a broader family of topological codes. The general design principle behind topological codes is that it is built by attaching repeated elements together.
The specific advantage of surface code is that it requires only nearest neighbour interaction. Many quantum computing platforms are unable to perform high-fidelity long-range interaction between qubits

#Four Cycle Code
This is the code for building the fundamental building block of surface codes i.e., the four-cycle code.
It is so called the four-cycle code because it consist of n=2 stabilizer and m=2 code qubits thus, it can encode k=n-m=0 logical qubits.
As a result, the four cycle is not in itself a useful code.
However, we could see that working detection and correction code can be formed by tiling together many four-cycle surface code.

```python
cq = QuantumRegister(2, 'D')
lq = QuantumRegister(2, 'A')
sb = ClassicalRegister(1, 'syndrome_bit')
qc = QuantumCircuit(cq, lq)
qc.h(lq[0])
qc.h(lq[1])
qc.cx(cq[0], lq[0])
qc.cx(cq[1], lq[0])
qc.cz(cq[0], lq[1])
qc.cz(cq[1], lq[1])
qc.h(lq[0])
qc.h(lq[1])
qc.draw()
```

#[5,1,2] Surface Code

```python
cq = QuantumRegister(5, 'D')
lq = QuantumRegister(4, 'A')
sb = ClassicalRegister(1, 'syndrome_bit')
qc = QuantumCircuit(cq, lq)
#for bringing it into superposition
qc.h(lq[0])
qc.h(lq[1])
qc.h(lq[2])
qc.h(lq[3])
#applying CNOT gates to different codequbits
#it will detect all the bit flip errors
qc.cx(cq[0], lq[0])
qc.cx(cq[1], lq[0])
qc.cx(cq[2], lq[0])
qc.cx(cq[3], lq[3])
qc.cx(cq[4], lq[3])
qc.cx(cq[2], lq[3])
#applying CZ gates to different code qubits
#it will detect all the phase flip errors
qc.cz(cq[0], lq[1])
qc.cz(cq[2], lq[1])
qc.cz(cq[3], lq[1])
qc.cz(cq[1], lq[2])
qc.cz(cq[4], lq[2])
qc.cz(cq[2], lq[2])
#bringing it back from superposition state
qc.h(lq[0])
qc.h(lq[1])
qc.h(lq[2])
qc.h(lq[3])

qc.draw()
```
