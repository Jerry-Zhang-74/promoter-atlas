# -*- coding: utf-8 -*-
# Minimal read-only UCSC 2bit reader (batch52, hg19 reference). Pure python, no deps.
# Format per UCSC Kent: header, index, per-seq (dnaSize,nBlocks,maskBlocks,packedDNA).
import struct, os

class TwoBit:
    def __init__(self, path):
        self.path = path
        self.fh = open(path, 'rb')
        sig = struct.unpack('<I', self.fh.read(4))[0]
        if sig == 0x1A412743:
            self.end = '<'
        elif sig == 0x4327411A:
            self.end = '>'
        else:
            raise ValueError('not a 2bit file: sig=0x%08X' % sig)
        ver, seqcount, res = struct.unpack(self.end + 'III', self.fh.read(12))
        if ver != 0:
            raise ValueError('2bit version != 0: %d' % ver)
        self.version = ver; self.seqcount = seqcount
        self.index = {}
        for _ in range(seqcount):
            nsz = self.fh.read(1)[0]
            name = self.fh.read(nsz).decode('ascii')
            off = struct.unpack(self.end + 'I', self.fh.read(4))[0]
            self.index[name] = off
        self._cache = {}

    def chrom_sizes(self):
        out = {}
        for name in self.index:
            self._load(name)
            out[name] = self._cache[name]['dnaSize']
        return out

    def _load(self, name):
        if name in self._cache:
            return
        off = self.index[name]
        self.fh.seek(off)
        dnaSize, nbc = struct.unpack(self.end + 'II', self.fh.read(8))
        nbs = struct.unpack(self.end + 'I'*nbc, self.fh.read(4*nbc)) if nbc else ()
        nsz = struct.unpack(self.end + 'I'*nbc, self.fh.read(4*nbc)) if nbc else ()
        mbc = struct.unpack(self.end + 'I', self.fh.read(4))[0]
        mbs = struct.unpack(self.end + 'I'*mbc, self.fh.read(4*mbc)) if mbc else ()
        msz = struct.unpack(self.end + 'I'*mbc, self.fh.read(4*mbc)) if mbc else ()
        self.fh.read(4)  # reserved
        dna_off = self.fh.tell()
        self._cache[name] = {'dnaSize': dnaSize, 'nbs': nbs, 'nsz': nsz,
                             'mbs': mbs, 'msz': msz, 'dna_off': dna_off}

    # 2 bit codes -> base; T=0 C=1 A=2 G=3
    _BASES = b'T C  A G '  # placeholder; use table below
    _LUT = bytes.maketrans(
        bytes([0b00000000,0b00000001,0b00000010,0b00000011,
               0b00000100,0b00000101,0b00000110,0b00000111,
               0b00001000,0b00001001,0b00001010,0b00001011,
               0b00001100,0b00001101,0b00001110,0b00001111]),
        b'TC TC TC TC '
    ) if False else None

    def sequence(self, name, start, end):
        """Return uppercase sequence string for 0-based half-open [start,end)."""
        self._load(name)
        c = self._cache[name]
        dnaSize = c['dnaSize']
        if start < 0 or end > dnaSize or start >= end:
            raise ValueError('range out of bounds %s:%d-%d (size %d)' % (name, start, end, dnaSize))
        # read packed bytes covering [start,end)
        b0 = start >> 2
        b1 = (end + 3) >> 2
        self.fh.seek(c['dna_off'] + b0)
        raw = self.fh.read(b1 - b0)
        # unpack
        table = bytes.maketrans(b'\x00\x01\x02\x03', b'T C A G'.replace(b' ', b''))  # placeholder
        # explicit unpack (fast enough)
        out = bytearray()
        code2base = b'TCAG'
        for byte in raw:
            out.append(code2base[(byte >> 6) & 3])
            out.append(code2base[(byte >> 4) & 3])
            out.append(code2base[(byte >> 2) & 3])
            out.append(code2base[byte & 3])
        seq = bytes(out[(start - (b0 << 2)):(start - (b0 << 2)) + (end - start)])
        # mask N blocks
        if c['nbs']:
            seqarr = bytearray(seq)
            for bstart, bsize in zip(c['nbs'], c['nsz']):
                bs, be = bstart, bstart + bsize
                if be <= start or bs >= end:
                    continue
                a = max(bs, start) - start
                b = min(be, end) - start
                for i in range(a, b):
                    seqarr[i] = ord('N')
            seq = bytes(seqarr)
        return seq.decode('ascii')

    def close(self):
        self.fh.close()

_COMP = bytes.maketrans(b'ACGTNacgtn', b'TGCANtgcan')
def rc(s):
    return s.encode('ascii').translate(_COMP)[::-1].decode('ascii')

if __name__ == '__main__':
    import sys
    tb = TwoBit(sys.argv[1])
    sz = tb.chrom_sizes()
    print('seqcount', tb.seqcount)
    for k in list(sz)[:30]:
        print(k, sz[k])
    if len(sys.argv) >= 5:
        print('probe', tb.sequence(sys.argv[2], int(sys.argv[3]), int(sys.argv[4])))
