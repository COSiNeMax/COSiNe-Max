import argparse
import random
import time
import numpy as np
import math
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import load_npz
from scipy.sparse import save_npz
from scipy.sparse.csgraph import connected_components

save = False

class Network():
    
    def __init__(self, t=50, target_prob=0.5, only_target=False, scc=False):
        self.partitions = np.load('partitions.npy')
        self.total_nodes = len(self.partitions)
        self.graph = None
        self.graph_type = 0
        self.load_transition_matrix()
        self.only_scc = scc
        self.targets = self.init_targets(target_prob)
        self.totals = self.init_totals(t+2, only_target)
        self.seeds = None
    
    def load_transition_matrix(self):
        self.graph = load_npz('transition-matrix.npz')
        self.graph_type = 1
    
    def load_adjacency_matrix(self):
        self.graph = load_npz('adjacency-matrix.npz')
        self.graph_type = 2
    
    def load_baseline3_transition_matrix(self):
        self.graph = load_npz('baseline3-transition-matrix.npz')
        self.graph_type = 3
    
    def init_targets(self, p):
        target = np.zeros(self.total_nodes)
        if not self.only_scc:
            for i in range(self.total_nodes):
                if random.random() < p:
                    target[i] = 1
        else:
            n_comp, labels = connected_components(self.graph, directed=True, connection='strong', return_labels=True)
            hist, bins = np.histogram(labels, bins=[i for i in range(max(labels)+2)])
            component = np.argmax(hist)
            self._comp_size = hist[component]
            for i in range(self.total_nodes):
                if labels[i] == component and random.random() < p:
                    target[i] = 1
        return target

    def init_totals(self, t, only_target):
        assert self.graph_type == 1
        totals = np.zeros(t)
        colors = np.copy(self.partitions)
        if only_target:
            colors *= self.targets
        self.graph = self.graph.tocsr()
        totals[0] = sum(Network.eval_stats(self.targets, self.partitions, colors).values())
        for i in range(1, t):
            colors = csr_matrix.dot(self.graph, colors)
            totals[i] = sum(Network.eval_stats(self.targets, self.partitions, colors).values())
        #print(totals)
        return totals

    def display_summary(self):
        #pos_nodes = (self.total_nodes + sum(self.partitions))/2#len([i for i in range(self.total_nodes) if self.partitions[i] > 0])
        pos_nodes = len([i for i in range(self.total_nodes) if self.partitions[i] > 0])
        print('partition +1 nodes: \t{}'.format(pos_nodes))
        print('partition -1 nodes: \t{}'.format(self.total_nodes - pos_nodes))
        print()
        pos_targets = sum([self.targets[i] for i in range(self.total_nodes) if self.partitions[i] > 0])
        print('targeted +1 nodes: \t{}'.format(pos_targets))
        print('targeted -1 nodes: \t{}'.format(sum(self.targets) - pos_targets))
        self.load_adjacency_matrix()
        self.graph = self.graph.tocoo()
        print()
        stats = {'P+ E+':0, 'p+ e-':0, 'P- E+':0, 'p- e-':0, 'ip e+':0, 'IP E-':0}
        wstats = {'P+ E+':0, 'p+ e-':0, 'P- E+':0, 'p- e-':0, 'ip e+':0, 'IP E-':0}
        for k in range(len(self.graph.data)):
            if self.partitions[self.graph.row[k]] == 1 and self.partitions[self.graph.col[k]] == 1:
                if self.graph.data[k] > 0:
                    stats['P+ E+'] += 1
                    wstats['P+ E+'] += self.graph.data[k]
                else:
                    stats['p+ e-'] += 1
                    wstats['p+ e-'] -= self.graph.data[k]
            elif self.partitions[self.graph.row[k]] == -1 and self.partitions[self.graph.col[k]] == -1:
                if self.graph.data[k] > 0:
                    stats['P- E+'] += 1
                    wstats['P- E+'] += self.graph.data[k]
                else:
                    stats['p- e-'] += 1
                    wstats['p- e-'] -= self.graph.data[k]
            else:
                if self.graph.data[k] > 0:
                    stats['ip e+'] += 1
                    wstats['ip e+'] += self.graph.data[k]
                else:
                    stats['IP E-'] += 1
                    wstats['IP E-'] -= self.graph.data[k]
        print('partition +1 edges: \t{}+ \t{}-'.format(stats['P+ E+'], stats['p+ e-']))
        print('partition -1 edges: \t{}+ \t{}-'.format(stats['P- E+'], stats['p- e-']))
        print('inter-partition edges: \t{}+ \t{}-'.format(stats['ip e+'], stats['IP E-']))
        print()
        print('partition +1 edge weights: \t{}+ \t{}-'.format(round(wstats['P+ E+'], 1), round(wstats['p+ e-'], 1)))
        print('partition -1 edge weights: \t{}+ \t{}-'.format(round(wstats['P- E+'], 1), round(wstats['p- e-'], 1)))
        print('inter-partition edge weights: \t{}+ \t{}-'.format(round(wstats['ip e+'], 1), round(wstats['IP E-'], 1)))
        self.load_transition_matrix()

    def eval_stats_purple(targets, partitions, colors):
        result = {'P+ C+':0.0, 'p- c+':0.0, 'p+ c-':0.0, 'P- C-':0.0}
        for i in range(len(targets)):
            if targets[i] == 1:# and colors[i] != 0:
                #if partitions[i] > 0:
                #    if colors[i] > 0:
                #        result['P+ C+'] += (1+colors[i])/2
                #    elif colors[i] <= 0:
                #        result['p+ c-'] += (1+colors[i])/2
                #    else:
                #        print('COLOR ERROR')
                #elif partitions[i] < 0:
                #    if colors[i] > 0:
                #        result['p- c+'] += (1-colors[i])/2
                #    elif colors[i] <= 0:
                #        result['P- C-'] += (1-colors[i])/2
                #    else:
                #        print('COLOR ERROR')
                #else:
                #    print('PARTITION ERROR')
                if partitions[i] > 0 and colors[i]>0:
                    result['P+ C+'] += (1+colors[i])/2
                elif partitions[i] < 0 and colors[i]<0:
                    result['P- C-'] += (1-colors[i])/2
                else:
                    pass#print('PARTITION ERROR')
        return result

    def eval_stats(targets, partitions, colors):
        result = {'P+ C+':0.0, 'p- c+':0.0, 'p+ c-':0.0, 'P- C-':0.0}
        for i in range(len(targets)):
            if targets[i] == 1 and colors[i] != 0:
                if partitions[i] > 0:
                    if colors[i] > 0:
                        result['P+ C+'] += colors[i]
                    elif colors[i] < 0:
                        result['p+ c-'] -= colors[i]
                    else:
                        print('COLOR ERROR')
                elif partitions[i] < 0:
                    if colors[i] > 0:
                        result['p- c+'] += colors[i]
                    elif colors[i] < 0:
                        result['P- C-'] -= colors[i]
                    else:
                        print('COLOR ERROR')
                else:
                    print('PARTITION ERROR')
        return result
    
    """
    def eval_stats_binarized(targets, partitions, colors, t):
        result = {'P+ C+':0.0, 'p- c+':0.0, 'p+ c-':0.0, 'P- C-':0.0}
        threshold = 1 / ((1*t)+1) #math.pow(3,-t)
        for i in range(len(targets)):
            if targets[i] == 1 and colors[i] != 0:
                if partitions[i] > 0:
                    if colors[i] >= threshold:
                        result['P+ C+'] += 1
                    elif colors[i] <= -threshold:
                        result['p+ c-'] -= -1
                    else:
                        pass#print('COLOR ERROR')
                elif partitions[i] < 0:
                    if colors[i] >= threshold:
                        result['p- c+'] += 1
                    elif colors[i] <= -threshold:
                        result['P- C-'] -= -1
                    else:
                        pass#print('COLOR ERROR')
                else:
                    print('PARTITION ERROR')
        return result
    """

    def evaluate(self, t, algo=''):
        assert self.seeds is not None
        assert self.graph_type == 1
        self.graph = self.graph.tocsr()
        colors = self.seeds
        #for i in range(self.total_nodes):
        #    if colors[i] == 0:
        #        colors[i] = random.choice([1,-1])
        for i in range(0, t):
            colors = csr_matrix.dot(self.graph, colors)
        if save:
            if self.only_scc:
                np.save('saved/epinions_{}_{}_{}_{}_scc.npy'.format(algo, t, sum(abs(self.seeds)), sum(self.targets)), colors)
            else:
                np.save('saved/epinions_{}_{}_{}_{}.npy'.format(algo, t, sum(abs(self.seeds)), sum(self.targets)), colors)
        res1 = Network.eval_stats_purple(self.targets, self.partitions, colors)
        res2 = Network.eval_stats(self.targets, self.partitions, colors)
        self.seeds = None
        result = (res1['P+ C+'] + res1['P- C-'], res2['P+ C+'] + res2['P- C-'])
        return result #- result['p+ c-'] - result['p- c+']

    def baseline1(self, t, two_k):
        assert self.seeds is None
        begin = time.time()
        seed_indices = random.sample([i for i in range(self.total_nodes) if self.targets[i] == 1], two_k)
        self.seeds = np.zeros(self.total_nodes)
        for i in range(two_k):
            if self.partitions[seed_indices[i]] < 0:
                self.seeds[seed_indices[i]] = -1
            else:
                self.seeds[seed_indices[i]] = 1
        end = time.time()
        #print('\nB1 ---------------')
        #print(self.seeds)
        return end-begin
    
    def baseline2(self, t, two_k):
        assert self.seeds is None
        self.load_adjacency_matrix()
        begin = time.time()

        scores = self.graph.sum(axis=0)
        scores = np.ravel(scores)
        l = [None] * self.total_nodes
        for i in range(self.total_nodes):
            l[i] = (i, scores[i])#, abs(scores[i]))
        cnt = 0
        total = 0
        #seeds = []
        self.seeds = np.zeros(self.total_nodes)
        for elem in sorted(l, key=lambda x: x[1], reverse=True):
            if self.targets[elem[0]] == 0:
                continue
            cnt += 1
            total += elem[1]
            #seeds.append(elem[0])
            if self.partitions[elem[0]] < 0:
                self.seeds[elem[0]] = -1
            elif self.partitions[elem[0]] > 0:
                self.seeds[elem[0]] = 1
            else:
                print('b2 error')
            if cnt >= two_k:
                break
        #for i in range(len(seeds)):
        #    if self.partitions[seeds[i]] < 0:
        #        self.seeds[seeds[i]] = -1
        #    else:
        #        self.seeds[seeds[i]] = 1
        end = time.time()
        #print('\nB2 ---------------')
        #print(self.seeds)
        self.load_transition_matrix()
        return end-begin
    
    def baseline3(self, t, two_k):
        assert self.seeds is None
        self.load_baseline3_transition_matrix()
        self.graph = self.graph.tocsc()
        begin = time.time()
        self.seeds = np.zeros(self.total_nodes)
        c = np.array(self.partitions, dtype='float64')
        for i in range(len(c)):
            if c[i] < 0:
                c[i] = 0
        c *= self.targets
        for i in range(t):
            c = csc_matrix.dot(c, self.graph)
        l = [None] * self.total_nodes
        for i in range(self.total_nodes):
            l[i] = (i, c[i])
        cnt = 0
        for elem in sorted(l, key=lambda x: x[1], reverse=True):
            if elem[1] <= 0:
                break
            #seeds.append(elem[0])
            self.seeds[elem[0]] = 1
            cnt += 1
            if cnt >= two_k/2:
                break

        c = np.array(self.partitions, dtype='float64')
        for i in range(len(c)):
            if c[i] > 0:
                c[i] = 0
            else:
                c[i] *= -1
        c *= self.targets
        for i in range(t):
            c = csc_matrix.dot(c, self.graph)
        l = [None] * self.total_nodes
        for i in range(self.total_nodes):
            l[i] = (i, c[i])
        cnt = 0
        for elem in sorted(l, key=lambda x: x[1], reverse=True):
            if elem[1] <= 0:
                break
            #seeds.append(-1*elem[0])
            self.seeds[elem[0]] = -1
            cnt += 1
            if cnt >= two_k/2:
                break

        #for i in range(two_k):
        #    if self.partitions[seeds[i]] < 0:
        #        self.seeds[seeds[i]] = -1
        #    else:
        #        self.seeds[seeds[i]] = 1
        end = time.time()
        #print('\nB3 ---------------')
        #print(self.seeds)
        self.load_transition_matrix()
        return end-begin
    
    def influ_max(self, t, two_k):
        assert self.seeds is None
        assert self.graph_type == 1
        self.graph = self.graph.tocsc()
        final_walk_distribution = np.array(self.partitions, dtype='float64')
        final_walk_distribution *= self.targets
        begin = time.time()
        for tm in range(t):
            final_walk_distribution = csc_matrix.dot(final_walk_distribution, self.graph)
        l = [None] * self.total_nodes
        for i in range(len(final_walk_distribution)):
            l[i] = (i, final_walk_distribution[i], abs(final_walk_distribution[i]))
        cnt = 0
        total = 0
        self.seeds = np.zeros(self.total_nodes)
        for elem in sorted(l, key=lambda x: x[2], reverse=True):
            if elem[1] < 0:
                total += -1*elem[1]
                self.seeds[elem[0]] = -1
            else:
                total += elem[1]
                self.seeds[elem[0]] = 1
            cnt += 1
            if cnt >= two_k:
                break
        end = time.time()
        #print(self.seeds)
        return end-begin
    
    def run_tests(self, t, n):
        times = [0.0]*4
        results1 = [0.0]*4
        results2 = [0.0]*4
        #print(self.totals[t])
        den1 = 100#self.totals[t]
        den2 = self.totals[t]#sum(self.targets)
        
        times[0] = self.baseline1(t, n)
        num = self.evaluate(t, 'b1')
        results1[0] = 100*num[0] / den1
        results2[0] = 100*num[1] / den2
        #print(self.evaluate(t))
        #self.seeds = None
        
        times[1] = self.baseline2(t, n)
        num = self.evaluate(t, 'b2')
        results1[1] = 100*num[0] / den1
        results2[1] = 100*num[1] / den2
        #print(self.evaluate(t))
        #self.seeds = None
        
        times[2] = self.baseline3(t, n)
        num = self.evaluate(t, 'b3')
        results1[2] = 100*num[0] / den1
        results2[2] = 100*num[1] / den2
        #print(self.evaluate(t))
        #self.seeds = None
        
        times[3] = self.influ_max(t, n)
        num = self.evaluate(t, 'im')
        results1[3] = 100*num[0] / den1
        results2[3] = 100*num[1] / den2
        #print(self.evaluate(t))
        #self.seeds = None
        
        for i in range(4):
            results1[i] = round(results1[i], 2)
            results2[i] = round(results2[i], 2)
            times[i] = round(times[i], 5)
        print('\t{},\t{},\t{},\t{},\t{},\t{},\t{},\t{},\t{},\t{},\t{},\t{},\t{},\t{},\t{}'.format(t,n,int(sum(self.targets)),results1[0],results1[1],results1[2],results1[3],results2[0],results2[1],results2[2],results2[3],round(times[0], 5),round(times[1], 5),round(times[2], 5),round(times[3], 5)))

def main():
    global save
    parser = argparse.ArgumentParser()
    parser.add_argument('--t', type=int, default=50)
    parser.add_argument('--k', type=int, default=None)
    parser.add_argument('--tkp', type=int, default=10)
    parser.add_argument('--target', type=int, default=100)
    parser.add_argument('--print-stats', action='store_true')#type=bool, default=False)
    parser.add_argument('--only-target', action='store_true')#type=bool, default=False)
    #parser.add_argument('--old-metric', action='store_true')#type=bool, default=False)
    parser.add_argument('--scc', action='store_true')#type=bool, default=False)
    parser.add_argument('--save-colors', action='store_true')#type=bool, default=False)
    args = parser.parse_args()
    random.seed(0)
    save = args.save_colors
    soc_net = Network(args.t, args.target/100, args.only_target, args.scc)
    n = 0
    if args.k:
        n = args.k#2*int(args.k)
    else:
        n = int(soc_net.total_nodes * args.tkp/100)
    if args.print_stats:
        print('=========================================================================')
        print('NETWORK SUMMARY STATISTICS:')
        soc_net.display_summary()
    if args.print_stats:
        print('=========================================================================')
        if soc_net.only_scc:
            print('[scc has only {} nodes]'.format(soc_net._comp_size))
        if not args.only_target:
            print('TOTAL INFLUENCE: (starting with nodes within and without the target)')
        else:
            print('TOTAL INFLUENCE: (ony including nodes in the target)')
        print(soc_net.totals)
    #for i in range(args.t+1):
    #    print('\t{}, \t{}'.format(i, soc_net.totals[i]))
    if args.print_stats:
        print('=========================================================================')
        print('PERFORMANCE:')
    #for i in range(args.t):
    #    soc_net.run_tests(i, n)
    soc_net.run_tests(args.t, n)
    if args.print_stats:
        print('=========================================================================')

if __name__ == '__main__':
    main()

