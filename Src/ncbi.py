import os
import sys
import cPickle


from Bio import Entrez, SeqIO


PATH_INPUT = os.path.join('..', 'Data', 'rfam_data_full.cPickle')
PATH_OUTPUT = os.path.join('..', 'Data', 'seq_data_300.csv') 
LENGTH = 300


def ncbi_subsec(name, start, stop):
    try:
        Entrez.email = 'vreinharz@gmail.com'
        handle = Entrez.efetch(db='nuccore', id=name, seq_start=start, seq_stop=stop,
                               rettype='gb', retmode='text')
        record = SeqIO.read(handle, 'genbank')
        handle.close()
    except ValueError:
        print "Values Error: ", name, start, stop
        return ''
    seq = str(record.seq)
    if start < 1:
        seq = '-'*(abs(start)+1) + seq
    if len(seq) < stop-start:
        seq += '-'*(stop-start-len(seq))
    return seq

def main():
    def m(x):
        d = {'A':0, 'C':1, 'G':2, 'U':3, 'N':4, '-':-1}
        return d[x]
    with open(PATH_INPUT) as f:
        data = cPickle.load(f)
    with open(PATH_OUTPUT, 'w') as f:
        for i, fam in enumerate(sorted(data.keys())):
            for name in data[fam]['sequences']:
                print fam, name
                id, pos = name.split('/')
                pos_l, pos_r = map(int, pos.split('-'))
                pos_l, pos_r, order = (pos_l, pos_r, 1) if pos_l < pos_r else (pos_r, pos_l, -1)
                l = pos_r - pos_l + 1
                pos_l = pos_l - (LENGTH - l)/2
                pos_r = pos_l + LENGTH-1
                seq = ncbi_subsec(id, pos_l, pos_r)[::order].replace('T', 'U')
                if seq:
                    #out.append(','.join(map(str, map(m, list(seq))) + [name, str(i)])) 
                    f.write(','.join(map(str, map(m, list(seq))) + [name, str(i)]) + '\n')
            #f.write('\n'.join(out))

if __name__ == '__main__':
    main()
