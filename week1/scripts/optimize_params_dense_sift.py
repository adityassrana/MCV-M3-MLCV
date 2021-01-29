from w1_gridsearch_pca_logreg import *
import csv

if __name__ == "__main__":

    columns = ['Descriptor', 'Spatial pyramid level', 'Codebook size', 'Step size',
              'PCA perc', 'Cross-val prec', 'Cross-val recall', 'Cross-val f1',
              'Test prec', 'Test recall', 'Test f1', 'Test acc', 'PCA Test acc', 'LDA Test acc']

    csv_filename = 'optimize_dense_sift.csv'

    #with open(csv_filename, 'a', newline='') as f:
    #    writer = csv.writer(f, delimiter=',')
    #    writer.writerow(columns)

    feat_des = 'sift'
    dense = True

    spatial_pyramid_level = [1]
    step_size = [15]

    k_values = [128]
    pca_percs = [0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]

    for pyramid_level in spatial_pyramid_level:
        for step in step_size:
            params_pre = set_params(feat_des=feat_des,dense=dense,
                                level=pyramid_level,step=step)

            train_des, D, test_des = get_descriptors_D(params_pre)

            for k in k_values:
                for pca_perc in pca_percs:
                    params = set_params(feat_des=feat_des,dense=dense,level=pyramid_level,
                                        k=k,step=step,pca_perc=pca_perc)
                    scores = run(train_des, D, test_des, params)

                    to_write = ['DENSE SIFT', pyramid_level, k, step, pca_perc, *scores]

                    #with open(csv_filename, 'a', newline='') as f:
                    #    writer = csv.writer(f, delimiter=',')
                    print(to_write)
