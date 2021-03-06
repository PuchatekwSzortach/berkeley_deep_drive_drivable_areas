"""
Module with invoke tasks
"""

import invoke


import net.invoke.analyze
import net.invoke.docker
import net.invoke.ml
import net.invoke.tests
import net.invoke.visualize

# Default invoke collection
ns = invoke.Collection()

# Add collections defined in other files
ns.add_collection(net.invoke.analyze)
ns.add_collection(net.invoke.docker)
ns.add_collection(net.invoke.ml)
ns.add_collection(net.invoke.tests)
ns.add_collection(net.invoke.visualize)
