# Simulates Markov chain approximation to AR(1) process, runs value
# function iteration, simulates neoclassical growth model for ECON 6130
# Markov chain approximation
# AR(1) process: y_t+1=Ky_t+e_t, e_t IID normal
# Parameters: K, mean of e_t, long-run variance of y_t, long-run mean of
# y_t, size of sample space

using Distributions
using LinearAlgebra
using StatsBase
using Plots


# 2.2 - Approximate 7point Markov Chain for AR(1) process
K=0.98
mean_e=0
LRvar_y=0.1
LRmean_y=0
size_space=7
# Implied variance of e_t
var_e=(1-K^2)*LRvar_y
# 1-d vector for sample space
start_space = LRmean_y-((size_space-1)/2)*sqrt(LRvar_y)
end_space = LRmean_y+((size_space-1)/2)*sqrt(LRvar_y)
sample_space=range(start_space, end_space, size_space)

normcdf(x) = cdf(Normal(),x)
# 2-d transition matrix, conditional probabilities found using Tauchen (1986)
trans_mat=zeros(size_space, size_space)
for i=1:size_space
    for j=1:size_space
        if j==1
            trans_mat[i,j]=normcdf((sample_space[j]-K*sample_space[i]+0.5*sqrt(LRvar_y))/sqrt(var_e))
        elseif j==size_space
            trans_mat[i,j]=1-normcdf((sample_space[j]-K*sample_space[i]-0.5*sqrt(LRvar_y))/sqrt(var_e))
        else
            trans_mat[i,j]=normcdf((sample_space[j]-K*sample_space[i]+0.5*sqrt(LRvar_y))/sqrt(var_e))-normcdf((sample_space[j]-K*sample_space[i]-0.5*sqrt(LRvar_y))/sqrt(var_e))
        end
    end
end

# Calculate stationary distribution of transition matrix
A=trans_mat'-I
A[size_space,:]=ones(1,size_space)
b=zeros(size_space,1)
b[size_space]=1
stationary=Array((A\b)')
# Simulate Markov Chain for T steps, with initial state distributed
# according to stationary distribution
T=2000
y_state=zeros(1,T)
y_val=zeros(1,T)
for i=1:T
    num=rand(1)[1]
    done=0
    j=0

    while done==0
        j=j+1
        if i==1
            if j==1
                if num<=stationary[j]
                    done=1
                end
                elseif sum(stationary[1:j-1])<num && num<=sum(stationary[1:j])
                    done=1
                end
            else
                if j==1
                if num<=trans_mat[Int(y_state[i-1]),j]
                    done=1
                end
                elseif sum(trans_mat[Int(y_state[i-1]),1:j-1])<num && num<=sum(trans_mat[Int(y_state[i-1]),1:j])
                    done=1
                end
        end
    end
    y_state[i]=j
    y_val[i]=sample_space[Int(y_state[i])]
end

# mean, variance, and serial correlation of Markov Chain
acf = autocor(y_val')
mean(acf)
mean(y_val)
var(y_val)


#Create plot
plot(Array(y_val'),
    # collect(1:T), 
    title = "Markov Chain Simulation, T=2000",
    label = "Policy Function Value",
    xlabel = "Time",
    ylabel = "y_t",
    legend=:bottomright)


# 2.3 - Value function iteration
# Parameter Values
alpha=0.35
beta=0.95
# 1-d capital grid
grid_size=100
grid_max=18
k_grid=range(0,grid_max,grid_size)
# 3-d array for value functons. Entry (1,j,k) evaluates current value function
# at capital level k_grid[j] and income shock sample_space(k). Entry
# (2,j,k) to store next iteration of value function.
# Initialize with value(i,j,k)=0 for all i,j,k.
value=zeros(2,grid_size,size_space)
# Tolerance for value function error
tol=10^(-6)
# Supremum distance between value function 1 and value function 2,
# initialized at 1
sup=1
# 3-d array for value of objective in iterations. Entry (i,j,k) gives value
# of objective if current-period capital is k_grid[i], current income shock
# is sample_space[j], and next-period capital is k_grid(k)
value_iter=zeros(grid_size,size_space,grid_size)
# Loop to compute value functions iteratively, continue until sup<tol




while sup>=tol
# Rename previous iteration's value function as base value function for
# current iteration
    value[1,:,:]=value[2,:,:]
    # Loop over current income shock values in sample_space
    for k=1:size_space
    # Loop over current capital values in k_grid
        for i=1:grid_size
        # Loop over next-period capital values in k_grid
            for j=1:grid_size
            # Check if feasibility is satisfied, capital strictly
            # positive (Inada conditions assumed)
                if 0<k_grid[j] && k_grid[j]<=exp(sample_space[k])*k_grid[i]^alpha
                # Calculate value of objective with current-period capital
                # k_grid[i], next-period capital k_grid[j], and
                # current-period shock sample_space(k)
                value_iter[i,k,j]=(log(exp(sample_space[k])*k_grid[i]^alpha-k_grid[j])+beta)*trans_mat[k,:]'*value[1,Int(j),:]
                # Set value to -Inf if feasibility violated
                else
                value_iter[i,k,j]=-Inf
                end
            end
            # Assign value(2,i,k) as maximum of value_iter(i,k,j) over j
            value[2,Int(i),k]=findmax(value_iter[Int(i),Int(k),:])[1]
        end
    end
        
    # Determine sup difference between value(2,i,k) and value(1,i,k)
    sup=findmax(findmax(abs.(value[2,:,:]-value[1,:,:]))[1])[1]
end
# 2-d arrays for policy function and index of maximizing capital value.
# Entry (i,j) is index/capital value that maximizes value(2,i,j)
policy_ind=zeros(grid_size,size_space)
policy=zeros(grid_size,size_space)
for i=1:grid_size
    for j=1:size_space
        z,index=findmax(value_iter[i,j,:])
        policy_ind[i,j]=findmin(index)[1]
        policy[i,j]=k_grid[findmin(index)[1]]
    end
end

# Plot value function for each shock in sample_space
plot(    # collect(1:T), 
title = "SP Policy Function",
xlabel = "K",
ylabel = "Value",
legend=:bottomright
)
for j=1:size_space
    plot!(k_grid[:],
    value[2,:,j]
    )
end
current()
