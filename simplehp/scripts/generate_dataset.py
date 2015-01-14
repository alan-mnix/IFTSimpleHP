# Authors: Giovani Chiachia <giovani.chiachia@gmail.com>
#
# License: BSD

import os
import optparse
import pprint

from scipy import misc
from scipy import io

import hyperopt

from simplehp.opt.base import datasets, learning_algos
from simplehp.opt.hp import objective, build_search_space
from simplehp.util.util import load_hp

speed_thresh = {'seconds': 2.0, 'elements': 4}

class WriteAdapter(object):
	def __init__(self, dataprovider, filename):
		super(WriteAdapter, self).__init__()
		self.provider = dataprovider
		self.filename = filename

	def protocol_imgs(self):
		return self.provider.protocol_imgs()
	
	def protocol_eval(self, algo, feat_set):

		X = feat_set
		Y = []

		if 'protocol_labels' in dir(self.provider):
			Y = self.provider.protocol_labels()

		dataset = dict(X = X, Y = Y)

		io.savemat(self.filename, dataset, do_compression=True)
		#print feat_set.shape
		#return self.provider.protocol_eval(algo, feat_set)

	def hp_imgs(self):
		return self.provider.hp_imgs()

	def hp_eval(self, algo, feat_set):
		return self.hp_eval(self, algo, feat_set)


def protocol_eval(dataset, dataset_path, hp_fname, host, port,
                  learning_algo, output_file):

    data_obj = dataset(dataset_path)

    data_obj = WriteAdapter(data_obj, output_file)

    # -- load hp file
    hp_space, trials, _ = load_hp(hp_fname, host, port)

    # -- best trial
    try:
        best_trial = trials.best_trial
    except Exception, e:
        raise ValueError('problem retrieving best trial: %s' % (e))

    dataset_info = {'data_obj': data_obj,
                    'fn_imgs': 'protocol_imgs',
                    'fn_eval': 'protocol_eval'}

    search_space = build_search_space(dataset_info,
                                      learning_algo,
                                      hp_space=hp_space,
                                      n_ok_trials=1000000,
                                      batched_lmap_speed_thresh=speed_thresh)

    ctrl = hyperopt.Ctrl(trials=trials, current_trial=best_trial)
    domain = hyperopt.Domain(objective, search_space)

    best_hps = hyperopt.base.spec_from_misc(best_trial['misc'])

    print 'Extract Features ...'

    domain.evaluate(best_hps, ctrl, attach_attachments=True)

    print 'Ok'


def get_optparser():

    dataset_options = ''
    for k in sorted(datasets.keys()):
      dataset_options +=  ("     %s - %s \n" % (k, datasets[k].__name__))

    usage = ("usage: %prog <DATASET> <DATASET_PATH> <OUTPUT_FILE>\n\n"
             "+ DATASET is an integer corresponding to the following supported "
             "datasets:\n" + dataset_options + '\n'
             "+ HP_FNAME is the pkl file containing the result of a previous "
             "hyperparameter optimization."
            )

    parser = optparse.OptionParser(usage=usage)

    learn_algo_default = learning_algos['default']
    learning_algos.pop('default', None)
    learn_algo_opts = ' OPTIONS=%s' % (learning_algos.keys())

    parser.add_option("--hp_fname", "-F",
                      default=None,
                      type="str",
                      metavar="STR",
                      help=("Pickle file created by optimization in serial"
                            "mode. [DEFAULT='%default']"
                            )
                        )

    parser.add_option("--host", "-H",
                      default=None,
                      type="str",
                      metavar="STR",
                      help=("Host serving MongoDB database created by "
                            "optimization running in asynchronous, parallel "
                            "mode. [DEFAULT='%default']"
                            )
                        )

    parser.add_option("--port", "-P",
                      default=10921,
                      type="int",
                      metavar="INT",
                      help=("MongoDB port at host serving the database. "
                            "[DEFAULT='%default']"
                            )
                        )


    return parser


def main():
    parser = get_optparser()
    opts, args = parser.parse_args()

    if len(args) != 3 or (opts.hp_fname is None and opts.host is None):
        parser.print_help()
    else:
        try:
            dataset = datasets[args[0]]
        except KeyError:
            raise ValueError('invalid dataset option')

        dataset_path = args[1]
	output_file = args[2]

        hp_fname = opts.hp_fname
        host = opts.host
        port = opts.port

	#dummy parameter unused for saving dataset
        learning_algo = learning_algos['svm_ova']


        protocol_eval(dataset, dataset_path, hp_fname, host, port,
                      learning_algo, output_file)

if __name__ == "__main__":
    main()
