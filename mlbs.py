import numpy as np
import itertools as it


class mlbs:
    def __init__(self, data, target_resolution, start_resolution, input_range):
        # for calculate_residual function
        self.original_data = np.array(data)
        self.sr = start_resolution
        self.tr = target_resolution
        self.di = len(self.sr) - 1
        self.fdi = self.sr[-1]
        self.data = data
        self.input_range = input_range

    def approximation(self):
        self.slbs = slbs(self.sr, self.data, self.input_range)
        # initial approximation
        self.slbs.linear_slbs()
        while (self.slbs.cls <= self.tr).all():
            # this operation will change self.data
            self.calculate_residual()
            self.slbs.refine_control_lattice()
            cls = np.array(self.slbs.cls)
            cls[0:-1] = cls[0:-1] - 2
            next_level_slbs = slbs(cls, self.data, self.input_range)
            next_level_slbs.approximation()
            self.slbs.control_lattice = self.slbs.control_lattice + next_level_slbs.control_lattice

    def calculate_residual(self):
        residual = []
        for i in self.original_data:
            pos = i[:self.di]
            v = self.slbs.interpolation(pos)
            res = i[self.di:] - v
            newitem = np.append(pos, res)
            residual.append(newitem)
        self.data = np.array(residual)


class slbs:
    def __init__(self, control_lattice_shape, data, input_range):
        self.cls = control_lattice_shape  # grid shape
        self.data = data
        self.il = input_range[0]
        self.ih = input_range[1]
        self.interval = (self.ih - self.il) / (self.cls[:-1] - 1)
        self.di = len(self.cls) - 1
        self.fdi = self.cls[-1]
        self.input_range = input_range
        # is 3 or 2?
        self.cls[0:-1] = self.cls[0:-1] + 2
        self.omega = np.zeros(self.cls)
        self.delta = np.zeros(self.cls)
        self.control_lattice = np.zeros(self.cls)

    def approximation(self):
        s = np.ones(self.di + 1, dtype=np.intp) * 4
        s[-1] = 1
        h = np.ones(self.di + 1, dtype=np.intp)
        h[-1] = self.fdi
        for e in self.data:
            base_point = np.floor(
                (e[0:self.di] - self.input_range[0]) / self.interval).astype(np.intp)
            t = e[0:self.di] / self.interval - base_point
            m = self.basic(t)
            coef = 1
            for a in m:
                coef = np.outer(coef, a).ravel()
            coef2 = coef ** 2
            coef2_sum = np.sum(coef2)
            coef2 = coef2.reshape(s)
            slices = [slice(i, i + 4) for i in base_point]
            # broadcast rule
            self.omega[slices] = self.omega[slices] + coef2
            coef3 = coef ** 3
            zc = e[self.di:]
            self.delta[slices] = self.delta[slices] + \
                coef3.reshape(s) * np.array(zc).reshape(h) / coef2_sum
        nzidx = np.nonzero(self.omega)
        self.control_lattice[nzidx] = self.delta[nzidx] / self.omega[nzidx]

    def refine_control_lattice(self):
        # more control lattice points
        new_cls = np.array(self.cls)
        new_cls[0:-1] = self.cls[0:-1] * 2 + 3
        self.interval = self.interval / 2
        wt = np.array([1 / 8, 4 / 8, 6 / 8, 4 / 8, 1 / 8])
        # self.control_lattice = np.zeros(self.cls)
        new_lattice = np.zeros(new_cls)
        wmatrix = np.array(list(it.product(wt, repeat=self.di)))
        wmshape = np.ones(self.di + 1, dtype=np.intp) * 5
        wmshape[-1] = self.di
        wmatrix = wmatrix.reshape(wmshape)
        wmatrix = np.multiply.reduce(wmatrix, self.di)
        t = tuple(self.cls[0:-1])
        idx = np.indices(t)
        idx = np.stack(idx, self.di).reshape(-1, self.di)
        for i in idx:
            new_lattice_idx = i * 2
            tmatrix = wmatrix[..., np.newaxis]
            cs = np.ones(self.di + 1, dtype=np.intp)
            cs[-1] = self.fdi
            # broadcast rule
            f = tmatrix * self.control_lattice[tuple(i)].reshape(cs)
            slices = [slice(s, s + 5) for s in new_lattice_idx]
            new_lattice[slices] = new_lattice[slices] + f
        tmp = (self.cls[0:-1] - 3) * 2 + 3 + 3
        new_cls = np.zeros_like(self.cls)
        new_cls[0:-1] = (self.cls[0:-1] - 3) * 2 + 3
        new_cls[-1] = self.fdi
        # refine complete update data member
        self.cls = new_cls
        new_slices = [slice(3, j) for j in tmp]
        # NOTE: This is not deep copy!!
        self.control_lattice = new_lattice[new_slices]

    def linear_slbs(self):
        coff = self.data[:, :self.di]
        coff = np.hstack([coff, np.ones((len(coff), 1))])
        b = self.data[:, self.di:]
        # fetch solutions
        m = np.linalg.lstsq(coff, b)[0]
        t = tuple(self.cls[0:-1])
        idx = np.indices(t)
        idx = np.stack(idx, self.di).reshape(-1, self.di)
        for i in idx:
            ti = i - 1
            ti = ti * self.interval + self.il 
            ti = np.append(ti, 1)[..., np.newaxis].T
            v = np.dot(ti, m)[0]
            self.control_lattice[tuple(i)] = v

    def interpolation(self, p):
        s = np.ones(self.di + 1, dtype=np.intp) * 4
        s[-1] = 1
        base_point = np.floor(
            (p - self.input_range[0]) / self.interval).astype(np.intp)
        t = p / self.interval - base_point
        m = self.basic(t)
        coef = 1
        for a in m:
            coef = np.outer(coef, a).ravel()
        coef = coef.reshape(s)
        slices = [slice(i, i + 4) for i in base_point]
        # broadcast rule
        f = self.control_lattice[slices] * coef
        retval = np.sum(f.reshape(-1, self.fdi), 0)
        return retval

    def basic(self, t):
        t2 = t * t
        t3 = t * t2
        b0 = 1 / 6 - t / 2 + t2 / 2 - t3 / 6
        b1 = t3 / 2 - t2 + 2 / 3
        b2 = -t3 / 2 + t2 / 2 + t / 2 + 1 / 6
        b3 = t3 / 6
        return np.array([b0, b1, b2, b3]).T
