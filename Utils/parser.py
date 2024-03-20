import argparse
import sys
import textwrap
import warnings
import numpy
from argparse import HelpFormatter as BaseHelpFormatter
from Utils.utils import to_paired_value


__version__ = "1.0.0"

warnings.filterwarnings("ignore")

class HelpFormatter(BaseHelpFormatter):
    def __init__(self, prog):
        super().__init__(prog=prog, max_help_position=45, width=120)
        
class Parser(argparse.ArgumentParser):
    def __init__(self, title=None, prefix=None, **kwargs):
        self.prefix = prefix
        self.title = title
        super().__init__(**kwargs)
    
    def format_usage(self):     
        formatter = HelpFormatter(prog=self.prog)
        formatter.add_text("{model}".format(model=self.description))
        formatter.add_usage(usage=self.usage, actions=self._actions,
                            groups=self._mutually_exclusive_groups, prefix=self.prefix)#"USAGE: -model "
        # description
        formatter.add_text("{model}'s ARGUMENT DESCRIPTIONS:".format(model=self.title))
        for action_group in self._action_groups:
            formatter.start_section(action_group.title)
            formatter.add_text(action_group.description)
            formatter.add_arguments(action_group._group_actions)
            formatter.end_section()
            
        return formatter.format_help()

class eMODiTSParameters(Parser):
            
    def __init__(self, commands: list):
        super().__init__(description='enhanced Multiobjective Symbolic Discretization for Time Series', prefix='USAGE: ', title='eMODiTS')
        self.available_commands = commands
        self.global_arguments = ['--version', '--help']
        self.subcommands = ["export"]
        self.add_argument(
            "-v",
            "--version",
            action="version",
            help="show program's version number and exit",
            version=__version__,
        )
        self.add_argument("-e", type=int, help="Executions number. Type of data: integer. Required argument.", required=False)
        self.add_argument("-g", type=int, help="Generations number. Type of data: integer. Required argument.",required=False)
        self.add_argument("-ps", type=int, help="Population size. Type of data: integer. Required argument.",required=False)
        self.add_argument("-pm", type=float, help="Mutation percentage. Type of data: double. Default value = %(default)s",default=0.2)
        self.add_argument("-pc", type=float, help="Crossover percentage. Type of data: double. Default value = %(default)s",default=0.8)
        self.add_argument("-ff", type=int, help="Fitness function configuration. 0: Entropy, complexity, infoloss. Type of data: integer. Default value = %(default)s",default=0)
        self.add_argument("-ids", type=int, nargs='+', help="IDs of the datasets, format: id1 id2 ... idn. Type of data: list of integer separated by a space. Required argument.", required=False)
        self.add_argument("-iu", type=float, help="Individuals' percentage to evaluate in original functions (individual-based strategy). Type of data: float. Default value = %(default)s", default=0.1)
        self.add_argument("-train-size-factor", type=int, help="Factor to calculate the size of the training set from the population size. Type of data: int. Default value = %(default)s", default=2)
        self.add_argument("-eval-method", type=str, help="Method used for evaluate new individuals: 'classic', 'surrogate'. Type of data: string. Default value = %(default)s", default='classic')
        self.add_argument("-task", type=str, help="Data mining task for the models. 'regression': Regression task, 'classification': Classification task. Type of data: string. Default value = %(default)s", default='regression')
        self.add_argument("-evaluation-measure", type=str, help="Metric used to evaluate the surrogate model. 'MSE' (Mean Square Error), 'R' (R Coefficient), 'R2' (R Squared), 'RMSE' (Root Mean Square Error), 'MD' (Modified Index of acceptance), 'MAPE' (Mean Absolute Percentage Error). Type of data: string. Default value = %(default)s", default='MD')
        self.add_argument("-train-rep", type=str, help="Representation type used for the surrogate model train set. 'all' = A Vector with all values, 'allnorm' = A normalized vector with all values, 'numcuts' = Vector with only number of cuts, 'stats' = Vector with stats values, 'cutdits' = Vector with cut distributions. Type of data: string. Default value = %(default)s",default="all")
        self.add_argument("-exec-type", type=str,help="Where is executed the program: 'cpu'. Type of data: string. Default value = %(default)s",default='cpu')
        self.add_argument("-error-t", type=float, help="Threshold error for updating the archive. Type of data: float. Default value = %(default)s", default=0.1)
        self.add_argument("-batch-update", type=int, help="Instances' number to add in each model to be updated. Type of data: int. Default value = %(default)s", default=1)
        self.add_argument("-checkpoints", help="Create checkpoints for restoring the program execution. ", action='store_true')
        self.add_argument("-no-checkpoints", help="Avoid the creation of checkpoints for restoring the program execution. ", dest='checkpoints', action='store_false')
        self.set_defaults(checkpoints=True)
        self.add_argument("-profilers", help="Create profiler files during the program execution.", action='store_true')
        self.add_argument("-no-profilers", help="Avoid the creation of checkpoints for restoring the program execution. ", dest='profilers', action='store_false')
        self.set_defaults(profilers=False)
        
        self.add_argument("-cache", help="Use the distance cache for speed up the runtime.", action='store_true')
        self.add_argument("-no-cache", help="Avoid the use the distance cache for speed up the runtime. ", dest='cache', action='store_false')
        self.set_defaults(cache=False)
        
        self.add_argument(
            "-model",
            choices=self._get_available_commands(),
            action = "append",
            help="List of surrogate models to use for each objective function. 'knn': KNNRegressor, 'rbf': RBF, 'svr': SVR, 'rbfnn': RBF Neural Network. Type of data: integer. Required argument.",
            required=False
        )
    
        #self._subparsers = self.add_subparsers(dest="command", parser_class=argparse.ArgumentParser)
        #export_parser = self._subparsers.add_parser("export", help="Export results of emodits executions.")
        #export_parser.add_argument("-dir", action = "append", help="Directories of the compared methods.", required = True)
        
    def _get_available_commands(self):
        return {cmd().prog for cmd in self.available_commands}

    def parse_args(self, *_):
        arg_strings = sys.argv[1:]
        #print("parser.parse_args.arg_strings:", arg_strings)
        arg_strings_iter = iter(arg_strings)
        parameter_models = {}
        params = []
        exists_models = False
        idx = [i for i in range(0,len(arg_strings)) if arg_strings[i].startswith('-model')]
        for i in range(0,len(idx)):
            exists_models = True
            parameter_models[i] = []
        
        idx.append(numpy.inf)
        model_indexs = to_paired_value(idx)
        
        exists_global = False
        for i, arg_string in enumerate(arg_strings_iter):
            if self.global_arguments.count(arg_string) <= 0 and self.subcommands.count(arg_string) <= 0:
                exists_global = False
                if '--' in arg_string: #parámetros de los modelos
                    idx_model = [j for j,n in enumerate(model_indexs) if n[0] < i < n[1]][0]
                    parameter_models[idx_model].append(arg_string)
                    parameter_models[idx_model].append(arg_strings[i+1])
                elif arg_string[0] == '-' and arg_string[1] != '-':
                    params.append(arg_string)
                    j = i+1
                    while arg_strings[j][0] != '-':
                        params.append(arg_strings[j])
                        j+=1
            else:
                exists_global = True
                params.append(arg_string)
        #print("parser.parser_args.params:", params)
        args = super().parse_args(params)
        if not exists_global:
            if exists_models:
                for i in range(0,len(args.model)):
                    subargs = self.parse(args.model[i], parameter_models[i])
                    getattr(args, 'model')[i] = subargs
        #print("parser.parse_args.args:",args) 
        #exit(1)
        return args

    def parse(self, cmd, model_options=None):
        for each in self.available_commands:
            if each().prog == cmd:
                parser_class = each
                description = each().description              
                break
        else:
            raise ValueError("No such commands: {}".format(cmd))
        
        #print("parser.parse.model_options:", model_options)

        parser = parser_class(
            prog="{} {}".format(self.prog, cmd),
            description=description,
            formatter_class=HelpFormatter,
            add_help=False
        )
        arguments = parser.parse_args(model_options)
        #print("parser.parse.arguments:",arguments)
        return arguments
    
    def print_help(self, file=None):
        if file is None:
            file = sys.stdout
        self._print_message(self.format_help(), file)
        
    def _get_hints(self):
        formatter = self._get_formatter()
        # description
        #formatter.add_text(self.description)
        # positionals, optionals and user-defined groups
        formatter.add_text("{model}' ARGUMENT DESCRIPTIONS:".format(model=self.prog.replace('.py','')))
        for action_group in self._action_groups:
            formatter.start_section(action_group.title)
            formatter.add_text(action_group.description)
            formatter.add_arguments(action_group._group_actions)
            formatter.end_section()
            
        # determine help from format above
        return formatter.format_help()

    def format_help(self):
        commands = ""
        for each in self.available_commands:                    
            commands += "{}\n".format(each().format_usage())

        help_msg = textwrap.dedent(
            """\
            {prog} {version}

            {usage}
            
            AVAILABLE MODELS:
            
            {commands}
            """
        ).format(
            version=__version__,
            prog=self.prog,
            usage=self.format_usage().replace('usage:',''),
            commands=commands,
            #helps=self._get_hints(),
        ).replace('options:','')

        return help_msg
