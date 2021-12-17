# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 11:11:15 2021

@author: mszx2
"""

from subprocess import check_output as qx
import os
from gensim.models import KeyedVectors
import random
from itertools import islice, chain


def generator_tor_hosts():
    cmd = os.path.join(os.getcwd(), r'generator\TorGenHost\TorGenHost.exe')
    print(cmd)
    buf = []
    def extract_hosts(out):
        res = out.decode().split('\r\n')[1:-1]
        return res
    while True:
        if len(buf) == 0:
            output = qx(cmd)
            buf.extend(extract_hosts(output))
        host = buf.pop()
        yield host

def generator_real_hosts():
    def extract_hosts(lines):
        return list(sorted(map(lambda line: line[:-1].lower(), lines), key = lambda k: random.random()))
    path = r'./hosts'
    buf = []
    onlyfiles = (os.path.join(path, f) for f in sorted(os.listdir(path), key = lambda k: random.random()) if 
    os.path.isfile(os.path.join(path, f)))
    while True:
        if len(buf) == 0:
            try:
                file = next(onlyfiles)
                with open(file, 'r') as f:
                    buf.extend(extract_hosts(f.readlines()))
            except:
                break
        yield 'www.' + buf.pop()

def cutter(generator):
    def cut(s):
        return s.split('.')[1]
    while True:
        try:
            res = next(generator)
            yield cut(res)
        except:
            break

def load_model(pretrained_model = "./wiki-news-300d-1M-subword.vec"):
    kv = KeyedVectors.load_word2vec_format(pretrained_model)
    return kv

def mix(gen1, gen2, buf_size = 10):
    buf = []
    while True:
        if len(buf) == 0:
            cond = random.random() > 0.5
            generator = gen1 if cond else gen2
            idx = 0 if cond else 1
            try:
                res = [*islice(generator, buf_size)]
                buf.extend(res)
            except:
                break
        yield buf.pop(), idx



if __name__ == '__main__':
    tor_hosts_generator = generator_tor_hosts()
    real_hosts_generator = generator_real_hosts()
    
    cutted_tor_hosts_generator = cutter(tor_hosts_generator)
    cutted_real_hosts_generator = cutter(real_hosts_generator)
    
    mixed_host_generator = mix(cutted_tor_hosts_generator,
                               cutted_real_hosts_generator)
    
    model = load_model()