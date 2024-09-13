import numpy as np
import galois
import math
import torch
import torch.nn as nn

GF256 = galois.GF(2**8)

# functions used for keypair generation and message signing
def generate_random_polynomial(o,v):
    f_i = np.vstack(
    (np.hstack((np.zeros((o,o),dtype=np.uint8),np.random.randint(2, size=(o,v), dtype=np.uint8))),
    np.random.randint(2, size=(v,v+o),dtype=np.uint8))
    )
    f_i_triu = np.triu(f_i)
    return GF256(f_i_triu)

def generate_central_map(o,v):
    F = []
    for _ in range(o):
        F.append(generate_random_polynomial(o,v))
    return F

def generate_affine_L(o,v):
    found = False
    while not found:
        try:
            L_n = np.random.randint(2, size=(o+v,o+v), dtype=np.uint8)
            L = GF256(L_n)
            L_inv = np.linalg.inv(L)
            found = True
        except:
            found = False
    return L, L_inv

def generate_random_vinegar(v):
    vv = np.random.randint(2, size=v, dtype=np.uint8)
    rvv = GF256(vv)
    return rvv

def sub_vinegar_aux(rvv,f,o,v):
    coeffs = GF256([0]* (o+1))
    # oil variables are in 0 <= i < o
    # vinegar variables are in o <= i < n 
    for i in range(o+v):
        for j in range(i,o+v):
            # by cases
            # oil and oil do not mix
            if i < o and j < o:
                pass
            # vinegar and vinegar contribute to a constant
            elif i >=o and j >= o:
                ij = GF256(f[i,j])
                vvi = GF256(rvv[i-o])
                vvj = GF256(rvv[j-o])
                coeffs[-1] += np.multiply(np.multiply(ij,vvi), vvj)
            # have mixed oil and vinegar variables that contribute to o_i coeff
            elif i < o and j >= o:
                ij = GF256(f[i,j])
                vvj = GF256(rvv[j-o])
                coeffs[i] += np.multiply(ij,vvj)
            # condition is not hit as we have covered all combos
            else:
                pass
    return coeffs

def sub_vinegar(rvv,F,o,v,attempts):
    subbed_rvv_F = []
    i = 0
    for f in F:
        i += 1
        print(f"Attempt {attempts}, iteration {i}/{len(F)}.",end="\r")
        subbed_rvv_F.append(sub_vinegar_aux(rvv,f,o,v))
    print()
    los = GF256(subbed_rvv_F)
    return los

# main functions
def generate_private_key(o,v): 
    F = generate_central_map(o,v)
    L, L_inv = generate_affine_L(o,v)
    return F, L, L_inv

def generate_public_key(F,L):
    L_T = np.transpose(L)
    P = []
    for f in F:
        s1 = np.matmul(L_T,f)
        s2 = np.matmul(s1,L)
        P.append(s2)
    return torch.from_numpy(GF256(P))

def sign(F,L_inv,o,v,message: torch.ByteTensor) -> torch.ByteTensor:
    signed = False
    attempts = 0
    m = GF256(message.unsqueeze(-1).detach().cpu().numpy())
    while not signed:
        try:
            attempts += 1
            rvv = generate_random_vinegar(v)
            los = sub_vinegar(rvv,F,o,v,attempts)
            M = GF256(los[:, :-1])
            c = GF256(los[:, [-1]])
            y = np.subtract(m,c)
            x = np.vstack((np.linalg.solve(M,y), rvv.reshape(v,1)))
            s = np.matmul(L_inv, x)
            signed = True
        except:
            signed = False
    return torch.from_numpy(s).squeeze()

#def verify(P,s,m):
#    cmparts= []
#    s_T = np.transpose(s)
#    for f_p in P:
#        cmp1 = np.matmul(s_T,f_p)
#        cmp2 = np.matmul(cmp1,s)
#        cmparts.append(cmp2[0])
#    computed_m = GF256(cmparts)
#    return computed_m, np.array_equal(computed_m,m)

def remainder_mod2(x):
    y = torch.pow(torch.sin(torch.mul(x, math.pi/2)), 2)
    return y - y.detach() + y.round().detach()

# create lookuptable for field of size n
def create_lookuptable(n: int):
    return n
    

class VerificationLayer(nn.Module):
    def __init__(self, P: torch.ByteTensor):
        super(VerificationLayer, self).__init__()
        self.P = P

    def forward(self, m: torch.ByteTensor, s: torch.ByteTensor):
        s = s.unsqueeze(-1)
        s_T = torch.transpose(s, -2, -1)
        # simple parity check: if all elements of element-wise subtracted m and computed m are 0 (mod 2), m and computed m are equal in the F2 field
        print(f"Now m_check is: {torch.squeeze(s_T @ self.P @ s)}")
        m_check = remainder_mod2(torch.squeeze(s_T @ self.P @ s) - m)
        # now compute ReLU(1 - sum(m_check)) to check if all elements of m_check are indeed 0: outputs 1 if they are all 0, outputs 0 if there is a 1
        return nn.functional.relu(m_check.sum().mul(-1).add(1))

# test a bunch of times for small parameters
def test():
    torch.manual_seed(2)
    messagelength = 16
    o = messagelength
    v = messagelength*2
    F, L, L_inv = generate_private_key(o,v)
    P = generate_public_key(F,L)
    verificationLayer = VerificationLayer(P)

    total_tests = 0
    tests_passed = 0

    for _ in range(10):
        total_tests+=1
        m = torch.randint(256, (messagelength,), dtype=torch.uint8)
        s = sign(F,L_inv,o,v,m)
        verified = verificationLayer(m, s)
        if verified.item() == 1.0:
            tests_passed+=1
        print(f"Test: {total_tests}\nMessage:\n{m}\nSignature:\n{s}\nVerified:\n{verified}\n")
    
    print(f"{tests_passed} out of {total_tests} messages verified.")

# illustrative example
def example():
    np.random.seed(0)
    messagelength = 16
    
    # field information

    print(f"{GF256.properties}\n")

    # parameters 

    o = messagelength
    v = messagelength*2
    print(f"The parameters are o={o} and v={v}.\n")

    # generate keys

    # private key
    F, L, L_inv = generate_private_key(o,v)
    print("Private Key:\n")

    print("Central Map:")
    for i,f in enumerate(F):
        print(f"{i}:\n {f}\n")
    
    print(f"Secret Transformation L=:\n{L}\n")
    print(f"Secret Inverse Transformation L_inv=:\n{L_inv}\n")
    print(f"Confirming L is invertible as L*L_inv is I=:\n {np.matmul(L,L_inv)}\n")

    # public key
    P = generate_public_key(F,L)
    print("Public Key = F ∘ L = \n")
    for i,f_p in enumerate(P):
        print(f"{i}:\n{f_p}\n")

    # message
    m = GF256([[x] for x in np.random.randint(256, size=messagelength, dtype=np.uint8)])
    print(f"Message m 128 random variables:\n{m}\n")

    # signing

    # generate random vinegar variables
    rvv = generate_random_vinegar(v)
    print(f"Random vinegar values:\n{rvv}\n")

    # sub vinegar variables
    los = sub_vinegar(rvv,F,o,v, 1)
    print(f"Substituted random vinegar variables:\n{los}\n")

    # subtract constant terms from message to get linear system
    M = GF256(los[:, :-1])
    c = GF256(los[:, [-1]])
    y = np.subtract(m,c)
    print("We separate out the constant terms of the linear oil system and subtract them from the message values to solve the linear system using Gaussian elimination.")
    print(f"y = m-c =\n {m} \n-\n {c}\n =\n {y}\n")
    print(f"f(o1,o2,o3) =\n {M}|{y}\n")
    # solve the linear system
    x_o = np.linalg.solve(M,y)
    x = np.vstack((x_o, rvv.reshape(v,1)))
    print(f"This yields the solution o1,o2,o3 =\n {x_o}\n")
    print(f"We stack this solution with our random vinegar variables to form a complete solution to the non-linear multivariate polynomial system of equations:\n {x}\n")
    
    # checking out solution
    print("We can check out solution by plugging them into the central map.")
    x_T = np.transpose(x)
    cmpartsc = []
    for f in F:
        cmcp1 = np.matmul(x_T,f)
        cmcp2 = np.matmul(cmcp1,x)
        cmpartsc.append(cmcp2[0])
    computed_mc = GF256(cmpartsc)
    print(f"m = x_T F x =\n {computed_mc}\n")

    # compute signature
    s = np.matmul(L_inv,x)
    print(f"Finally we compute our signature as:")
    print(f"sig = L_inv * x =\n {s}\n")

    # verification
    print("Now let's see if our signature is correct given our public_key and message:")
    verificationLayer = VerificationLayer(P)

    m = torch.from_numpy(m).flatten()
    s = torch.from_numpy(s).flatten()
    verified = verificationLayer(m, s)

    print(f"computed message == message is {verified.item() == 1}")

if __name__ == "__main__":
    test()
    #example()
