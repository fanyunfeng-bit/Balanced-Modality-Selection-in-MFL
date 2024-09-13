from config.utils import get_args
from server.mmfedavg import FedAvgServer
from server.mmfedcmd import FedCMDServer
from server.mmscaffold import SCAFFOLDServer
from server.mmfedprox import FedProxServer
from server.mmfedmi import FedMIServer
from server.FedCMI import FedCMIServer
from server.mmdrop import FedMDropServer
from server.pow_d import PowDServer
from server.agm import AGMServer


if __name__ == "__main__":
    args = get_args()

    if args.fl_method == 'FedAvg':
        server = FedAvgServer()
    elif args.fl_method == 'SCAFFOLD':
        server = SCAFFOLDServer()
    elif args.fl_method == 'FedProx':
        server = FedProxServer()
    elif args.fl_method == 'FedMI':
        server = FedMIServer()
    elif args.fl_method == 'FedCMD':
        server = FedCMDServer()
    elif args.fl_method == 'FedCMI':
        server = FedCMIServer()
    elif args.fl_method == 'FedMDrop':
        server = FedMDropServer()
    elif args.fl_method == 'PowD':
        server = PowDServer()
    elif args.fl_method == 'AGM':
        server = AGMServer(args, 'AGM')
    else:
        raise ValueError('No such fl method.')
    server.run()
