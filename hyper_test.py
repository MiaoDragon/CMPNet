from cmpnet_test import main
import argparse


parser = argparse.ArgumentParser()
# for training
parser.add_argument('--model_path', type=str, default='./models/',help='path for saving trained models')
parser.add_argument('--seen_N', type=int, default=0)
parser.add_argument('--seen_NP', type=int, default=0)
parser.add_argument('--seen_s', type=int, default=0)
parser.add_argument('--seen_sp', type=int, default=0)
parser.add_argument('--unseen_N', type=int, default=0)
parser.add_argument('--unseen_NP', type=int, default=0)
parser.add_argument('--unseen_s', type=int, default=0)
parser.add_argument('--unseen_sp', type=int, default=0)
parser.add_argument('--grad_step', type=int, default=1, help='number of gradient steps in continual learning')
# for continual learning
parser.add_argument('--n_tasks', type=int, default=1,help='number of tasks')
parser.add_argument('--n_memories', type=int, default=256, help='number of memories for each task')
parser.add_argument('--memory_strength', type=float, default=0.5, help='memory strength (meaning depends on memory)')
# Model parameters
parser.add_argument('--total_input_size', type=int, default=2800+4, help='dimension of total input')
parser.add_argument('--AE_input_size', type=int, default=2800, help='dimension of input to AE')
parser.add_argument('--mlp_input_size', type=int , default=28+4, help='dimension of the input vector')
parser.add_argument('--output_size', type=int , default=2, help='dimension of the input vector')
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--device', type=int, default=0, help='cuda device')
parser.add_argument('--data_path', type=str, default='../data/simple/')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--memory_type', type=str, default='res', help='res for reservoid, rand for random sampling')
parser.add_argument('--env_type', type=str, default='s2d', help='s2d for simple 2d, c2d for complex 2d')
parser.add_argument('--world_size', nargs='+', type=float, default=20., help='boundary of world')
parser.add_argument('--opt', type=str, default='Adagrad')
parser.add_argument('--train_path', type=int, default=1)
args = parser.parse_args()
model_path = args.model_path
for lr in [0.0001, 0.001, 0.01, 0.1, 1., 10.]:
    for train_path in [1, 2, 4, 8, 16, 32]:
        args.model_path = model_path + str(lr) + '/' + str(train_path) + '/'
        args.learning_rate = lr
        args.train_path = train_path
        main(args)
