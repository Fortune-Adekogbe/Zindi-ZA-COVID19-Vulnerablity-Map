from sklearn.base import BaseEstimator, TransformerMixin


class MCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None, method='indicator', 
                 benzecri_correction=True):
        self.benzecri_correction = benzecri_correction
        self.method = method
        self.n_components = n_components

    @property
    def method(self):
        """
        Matrix to do computations on `{'indicator', 'burt'}`
        """
        return self._method

    @method.setter
    def method(self, method):
        allowed = ['burt', 'indicator']
        if method not in allowed:
            raise TypeError(allowed)
        self._method = method

    def fit(self, X, y=None):
        """
        ``X`` should be a DataFrame of Categoricals.
        """
        df = X.copy()
        dummies = pd.get_dummies(df)
        Z = dummies.to_numpy()
        self.I_, self.Q_ = df.shape

        if self.method == 'indicator':
            C = Z
        elif self.method == 'burt':
            C = Z.T @ Z
        else:
            raise TypeError

        self.C_ = C
        Q = self.Q_
        J = Z.shape[1]
        N = self.n_components if self.n_components is not None else J - Q

        P = C / C.sum()
        cm = P.sum(0)
        rm = P.sum(1)
        eP = np.outer(rm, cm)
        
        self.test_ = stats.chisquare(P, eP, axis=None)
        S = (P - eP) / (np.sqrt(eP) + 1e-22) # add jitter

        u, s, v = np.linalg.svd(S, full_matrices=False)

        lam = s[:N]**2
        expl = lam / lam.sum()

        b = (v / np.sqrt(cm))[:N]                       # colcoord
        g = (b.T * np.sqrt(lam)).T                      # colpcoord

        u_red = u[:, :N]

        f = ((u_red * np.sqrt(lam)).T / np.sqrt(rm)).T  # rowcoord
        a = f / np.sqrt(lam)                            # rowpcoord

        # TODO: nicer names for these
        self.u_ = u
        self.s_ = s
        self.v_ = v
        self.b_ = b
        self.g_ = g
        self.explained_variance_ratio_ = expl
        self.J_ = J
        self.Z_ = Z
        self.P_ = P
        self.cm_ = cm
        self.rm_ = rm
        self.lam_ = lam
        self.f_ = f
        self.a_ = a
        self.names_ = dummies.columns
        return self

    def transform(self, X, y=None):
        return pd.get_dummies(X).values @ self.v_[:, :self.n_components]

    @staticmethod
    def adjust_inertia(σ, Q):
        σ_ = σ.copy()
        mask = σ_ >= 1 / Q
        σ_[mask] = ((Q / (Q - 1)) * (σ_[mask] - 1 / Q)) ** 2
        σ[~mask] = 0
        return σ_