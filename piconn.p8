function create_nn(layers)
 nn={}
 nn.n_layers=#layers
 nn.n_input=layers[1]
 nn.n_output=layers[#layers]
 nn.n_nodes=layers
 --layer values for fwd
 --no nn.layer[1] bec. input
 nn.node={}
 for l=1,#layers do
  nn.node[l]={}
  --bias
  nn.node[l][0]=1
 end
 --layer weights
 --nn.weight[l][j][k]:
 --wt from node j in l to k in l+1
 nn.weight={}
 for l=1,#layers-1 do
  nn.weight[l]={}
   for i=0,layers[l] do
    nn.weight[l][i]={}
    for j=1,layers[l+1] do
     nn.weight[l][i][j]=rnd()
    end
   end
 end
 --deltas for back propagation
 nn.delta={}
 for l=2,#layers do
  nn.delta[l]={}
 end
 return nn
end

--o: output, t: target
function sq_err(o,t)
 return 0.5 * (o-t)^2
end

function der_sq_err(o,t)
 return o-t
end

function err(o,t)
 return sq_err(o,t)
end

function der_err(o,t)
 return der_sq_err(o,t)
end


function total_error(output,target)
 error=0
 for i=1,min(#output,#target) do
  error+= err(output[i],target[i])
 end
 return error
end

function sig(x)
 return 1/(1+2.7083^(-x))
end

function der_sig(y)
 return y*(1-y)
end

function tanh(x)
 return (2.7083^x-2.7083^(-x))/(2.7083^x+2.7083^(-x))
end

function der_tanh(y)
 return 1-y^2
end

function act(x)
 return tanh(x)
end

function der_act(y)
 return der_tanh(y)
end

function fwd(nn,input)
 --setup input layer
 for i=1,nn.n_nodes[1] do
  nn.node[1][i]=input[i]
 end
 --handle other layers
 for l=1,nn.n_layers-1 do
  for j=1,nn.n_nodes[l+1] do
   nn.node[l+1][j]=0
   for i=0,nn.n_nodes[l] do
    nn.node[l+1][j]+=nn.node[l][i]*nn.weight[l][i][j]
   end
   nn.node[l+1][j]=act(nn.node[l+1][j])
  end
 end
end

function bwd(nn,input,output,rate)
 --find deltas
 --last layer
 local l_l=nn.n_layers
 for i=1,nn.n_nodes[l_l] do
  nn.delta[l_l][i]=der_err(nn.node[l_l][i],output[i])*der_act(nn.node[l_l][i])
 end
 --previous layers
 for l=l_l-1,2,-1 do
  for i=0,nn.n_nodes[l] do
   nn.delta[l][i]=0
   for j=1,nn.n_nodes[l+1] do
    nn.delta[l][i]+=nn.weight[l][i][j]*nn.delta[l+1][j]
   end
   nn.delta[l][i]*=der_act(nn.node[l][i])
  end
 end
 --update weights
 for l=1,l_l-1 do
  for i=0,nn.n_nodes[l] do
   for j=1,nn.n_nodes[l+1] do
    nn.weight[l][i][j]-=rate*nn.node[l][i]*nn.delta[l+1][j]
   end
  end
 end
end

function train(nn,input,output,rate)
 fwd(nn,input)
 bwd(nn,input,output,rate)
end









--------------------------------
--    test: learn addition    --
--------------------------------

cls(6)

-- create neural network
nn=create_nn({2,2,1})

-- screen buffer
buffer={}
for x=0,20 do
 buffer[x]={}
end

-- main loop
for e=1,1000 do

rectfill(2,2,127,7,6)
print("epoch: "..e,2,2,8)

print("goal: ",2,8,8)
for x=0,20 do
 for y=0,20 do
  z=(x+y)/40
  pset(x+54,y+14,z*16)
 end
end

print("output:",2,36,8)

-- train
for x=0,20 do
for y=0,20 do
 train(nn,{x,y},{(x+y)/(2*20)},0.01)
end
end

for x=0,20 do
 for y=0,20 do
  fwd(nn,{x,y})
  buffer[x][y]=nn.node[3][1]*16
 end
end

for x=0,20 do
 for y=0,20 do
  pset(x+54,y+42,buffer[x][y])
 end
end
end
