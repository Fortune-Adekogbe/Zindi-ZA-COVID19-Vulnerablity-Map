#%%
import pandas as pd
import numpy as np
import holoviews as hv
import hvplot.pandas

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer
from sklearn.metrics import get_scorer

from za_covid_map.linear_model import TweedieGLM
from za_covid_map.mca import MCA

hv.extension('bokeh')


# %%
train = context.catalog.load('train_maskedv2')
test = context.catalog.load('test_maskedv2')
descriptions = context.catalog.load('variable_descriptions_v2')
sample_submission = context.catalog.load('samplesubmissionv2')

#%%
train.target_pct_vunerable.hvplot.kde()

# %%
transformer = Pipeline([('poly', PolynomialFeatures()),
                        ('scale', StandardScaler()),
                        ('pca', PCA(100)),
                        ('rescale', StandardScaler())])
model = Pipeline([('transformer', transformer),
                  ('model', TweedieGLM(power=0, max_iter=1000))])
link = Pipeline([('function', FunctionTransformer())])
scorer = get_scorer('neg_root_mean_squared_error')

pipeline = TransformedTargetRegressor(regressor=model, transformer=link)



#%%
params = {'regressor__model__power': [0, 2, 3],
          'regressor__model__alpha': [0, 1e-5, 1e-3, 1e-1, 1]}
search = RandomizedSearchCV(pipeline, params, scoring=scorer)

# %%
X_train, y_train = train.drop(columns=['target_pct_vunerable', 'ward']), train.target_pct_vunerable.add(1e-9)

search.fit(X_train, y_train)

results = pd.DataFrame(search.cv_results_)
context.io.save('searchresults', results)


# %%
X_test = test.loc[:, X_train.columns]

# %%
# predict and plot
y_pred = pd.Series(search.predict(X_test), name = y_train.name)
context.io.save('submissionkde', y_pred.hvplot.kde())



# %%
# format submission
submission = sample_submission
submission.target_pct_vunerable = y_pred

context.io.save('submission', submission)