# -*- coding: utf-8 -*-
# batch52: indexed random-access reader for plain FASTA (Ensembl GRCh37 = hg19).
import mmap, os, json, pickle

class Faidx:
    def __init__(self, fa_path, index_path=None):
        self.path = fa_path
        self.index_path = index_path or (fa_path + ".bidx")
        self.f = open(fa_path, "rb")
        if os.path.exists(self.index_path):
            self.index = pickle.load(open(self.index_path, "rb"))
        else:
            self.index = self._build()
            pickle.dump(self.index, open(self.index_path, "wb"))
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)

    def _build(self):
        mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        n = len(mm)
        heads = []
        i = mm.find(b">")
        while i != -1:
            if i == 0 or mm[i-1] == 0x0A:
                heads.append(i)
            i = mm.find(b">", i+1)
        idx = {}
        for k, h in enumerate(heads):
            j = mm.find(b"\n", h)
            name = mm[h+1:j].split(b" ")[0].split(b"\t")[0].decode("ascii")
            base_off = j + 1
            # first line width
            e1 = mm.find(b"\n", base_off)
            line_bases = e1 - base_off
            lw = line_bases + 1
            h2 = heads[k+1] if k+1 < len(heads) else n
            nbytes = h2 - base_off
            full = nbytes // lw; rem = nbytes % lw
            length = full*line_bases + (rem-1 if rem > 0 else 0)
            idx[name] = (base_off, line_bases, lw, length)
        mm.close()
        return idx

    def size(self, name):
        return self.index[name][3]

    def fetch(self, name, start, end):
        base_off, lb, lw, length = self.index[name]
        if start < 0 or end > length or start >= end:
            raise ValueError("range %s:%d-%d out of 0..%d" % (name, start, end, length))
        # read from start's line to end's line
        l0 = start // lb; l1 = (end - 1) // lb
        b0 = base_off + l0*lw
        b1 = base_off + l1*lw + lb
        raw = self.mm[b0:b1]
        # strip newlines
        seq = raw.replace(b"\n", b"")
        off = start - l0*lb
        return seq[off:off + (end - start)].decode("ascii")

if __name__ == "__main__":
    import sys
    fx = Faidx(sys.argv[1])
    for k, v in list(fx.index.items()):
        print(k, v[3])
