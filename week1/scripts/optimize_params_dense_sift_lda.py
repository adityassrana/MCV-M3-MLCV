from w1_gridsearch_lda import *
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
    n_components = [1,2,3,4,5,6,7]

    for pyramid_level in spatial_pyramid_level:
        for step in step_size:
            params_pre = set_params(feat_des=feat_des,dense=dense,
                                level=pyramid_level,step=step)

            train_des, D, test_des = get_descriptors_D(params_pre)

            for k in k_values:
                for n_comp in n_components:
                    params = set_params(feat_des=feat_des,dense=dense,level=pyramid_level,
                                        k=k,step=step,lda_ncomponents=n_comp)
                    scores = run(train_des, D, test_des, params)

                    to_write = ['DENSE SIFT', pyramid_level, k, step, n_comp, *scores]

                    #with open(csv_filename, 'a', newline='') as f:
                    #    writer = csv.writer(f, delimiter=',')
                    print(to_write)
