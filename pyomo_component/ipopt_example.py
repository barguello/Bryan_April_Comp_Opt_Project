# A Suffix example for ipopt.
# Translated to Pyomo from AMPL model source at:
#   https://projects.coin-or.org/Ipopt/wiki/IpoptAddFeatures
#
# This Pyomo example is formulated as a python script.
# To run this script execute the following command:
#
# $ coopr_python ipopt_example.py
#
# Execution of this script requires that the ipopt
# solver is in the current search path for executables
# on this system. This example was tested using Ipopt
# version 3.10.2

from coopr.pyomo import *
from coopr.opt import SolverFactory
import coopr.environ
import scipy
import numpy
import matplotlib

### Create the ipopt solver plugin using the ASL interface
solver = 'ipopt'
solver_io = 'nl'
stream_solver = False    # True prints solver output to screen
keepfiles =     False    # True prints intermediate file names (.nl,.sol,...) 
opt = SolverFactory(solver,solver_io=solver_io)
###

### Create the example model
model = ConcreteModel()
model.x1 = Var(bounds=(1,5),initialize=1.0)
model.x2 = Var(bounds=(1,5),initialize=5.0)
model.x3 = Var(bounds=(1,5),initialize=5.0)
model.x4 = Var(bounds=(1,5),initialize=1.0)
model.obj = Objective(expr=model.x1*model.x4*(model.x1+model.x2+model.x3) + model.x3)
model.inequality = Constraint(expr=model.x1*model.x2*model.x3*model.x4 >= 25.0)
model.equality = Constraint(expr=model.x1**2 + model.x2**2 + model.x3**2 + model.x4**2 == 40.0)
###

### Declare all suffixes 
# Incoming Ipopt bound multipliers (obtained from solution)
model.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
model.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
# Outgoing Ipopt bound multipliers (sent to solver)
model.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
model.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)
# Obtain dual solutions from first solve and send to warm start
model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
###

### Generate the constraint expression trees if necessary
if solver_io != 'nl':
    # only required when not using the ASL interface
    model.preprocess()
###

### Send the model to ipopt and collect the solution
print 
print "INITIAL SOLVE"
results = opt.solve(model,tee=stream_solver)
# load the results (including any values for previously declared
# IMPORT / IMPORT_EXPORT Suffix components)
model.load(results)
###

### Print Solution
print "   %7s %12s %12s" % ("Value","ipopt_zL_out","ipopt_zU_out")
for v in [model.x1,model.x2,model.x3,model.x4]:
    print "%s %7g %12g %12g" % ( v,
                                 value(v),
                                 model.ipopt_zL_out.getValue(v),
                                 model.ipopt_zU_out.getValue(v) )
print "inequality.dual =", model.dual.getValue(model.inequality)
print "equality.dual   =", model.dual.getValue(model.equality)
###


### Set Ipopt options for warm-start
# The current values on the ipopt_zU_out and
# ipopt_zL_out suffixes will be used as initial 
# conditions for the bound multipliers to solve
# the new problem
for var in [model.x1,model.x2,model.x3,model.x4]:
    model.ipopt_zL_in.setValue(var,model.ipopt_zL_out.getValue(var))
    model.ipopt_zU_in.setValue(var,model.ipopt_zU_out.getValue(var))
opt.options['warm_start_init_point'] = 'yes'
opt.options['warm_start_bound_push'] = 1e-6
opt.options['warm_start_mult_bound_push'] = 1e-6
opt.options['mu_init'] = 1e-6
###

### Send the model to ipopt and collect the solution
print 
print "WARM-STARTED SOLVE"
results = opt.solve(model,tee=stream_solver)
# load the results (including any values for previously declared
# IMPORT / IMPORT_EXPORT Suffix components)
model.load(results)
###

### Print Solution
print "   %7s %12s %12s" % ("Value","ipopt_zL_out","ipopt_zU_out")
for v in [model.x1,model.x2,model.x3,model.x4]:
    print "%s %7g %12g %12g" % ( v,
                                 value(v),
                                 model.ipopt_zL_out.getValue(v),
                                 model.ipopt_zU_out.getValue(v) )
print "inequality.dual =", model.dual.getValue(model.inequality)
print "equality.dual   =", model.dual.getValue(model.equality)
###
