import numpy as np
def angle(X_output_layer,X_pca_new):
    layer_angle=[]
    for layer in range(len(X_pca_new)):
        layer_angle.append(np.degrees(np.arccos(np.diagonal(np.dot(X_output_layer[layer],np.transpose(X_pca_new[layer])))/(np.linalg.norm(X_output_layer[layer],axis=1)* np.linalg.norm(X_pca_new[layer],axis=1)))))
    return np.array(layer_angle)

def angle_layer(X_output_layer,X_pca_new):
    layer_angle=np.degrees(np.arccos(np.diagonal(np.dot(X_output_layer,np.transpose(X_pca_new)))/(np.linalg.norm(X_output_layer,axis=1)* np.linalg.norm(X_pca_new,axis=1))))
    return np.array(layer_angle)


# taking the angles wrt all the class projection and assigning the class to the image having the least angle  
def least_class(class_angle,num_images,number_class=10):
    temp_final=[]
    for layer in range(class_angle.shape[0]):
        temp=[[200000,200000] for _ in range(num_images)]
        for j in range(0,num_images):
            for k in range(0,number_class):
                num=k*num_images+j
                if class_angle[layer][num]<temp[j][1]:
                    temp[j]=[k,class_angle[layer][num]]
        temp_final.append(temp) 
    return temp_final

def least_class_layer(class_angle,num_images,number_class=10):
    temp=[[200000,200000] for _ in range(num_images)]
    for j in range(0,num_images):
        for k in range(0,number_class):
            num=k*num_images+j
            if class_angle[num]<temp[j][1]:
                temp[j]=[k,class_angle[num]]
    return temp
# without sending it to the network asking the accuracy.
def accuracy_angle(y_pred,y):
    score_f=[]
    for layer in range(len(y_pred)):
        score_l =0
        for image_i in range(len(y)):
            if y_pred[layer][image_i][0]==y[image_i]:
                score_l=score_l+1
        score_f.append(round(score_l/len(y),4))
    return score_f

def accuracy_angle_layer(y_pred,y):
    score_l =0
    for image_i in range(len(y)):
        if y_pred[image_i][0]==y[image_i]:
            score_l=score_l+1
    score_l=round(score_l/len(y),4)
    return score_l

# without sending it to the network asking the accuracy.
def acc_class_angle(y_pred,y,number_class=10):
    score_f=[]
    for class_i in range(0,number_class,1):   
        score_c=[]
        for layer in range(len(y_pred)):
            score_l=0
            count=0
            for image_i in range(len(y)):
                if y[image_i]==class_i:
                    count=count+1
                    if y_pred[layer][image_i][0]==y[image_i]:
                        score_l=score_l+1
            score_c.append(round(score_l/count,4))
        score_f.append(score_c)
    return score_f

def acc_class_angle_layer(y_pred,y,number_class=10):
    score_f=[]
    for class_i in range(0,number_class,1):   
        score_l=0
        count=0
        for image_i in range(len(y)):
            if y[image_i]==class_i:
                count=count+1
                if y_pred[image_i][0]==y[image_i]:
                    score_l=score_l+1
        score_l=round(score_l/count,4)
        score_f.append(score_l)
    return score_f


def plot_angle_layer(path,layer_angle,figname):

    for class_n in range(0,10,1):
        fig = plt.figure(figsize=(35, 3))
        for layer in range(angle_proj[class_n].shape[0]):
            fig.add_subplot(1,11,layer+1)
            plt.hist(angle_proj[class_n][layer],alpha=0.25,color=['b'],bins=20,label=str(layer+1)+' layer')
            plt.legend()
        plt.savefig(f'{path}/plotdis/class-{class_n}{figname}.jpeg') 
        
def mean_var_angle_layer(path,layer_angle,mean_var_name,number_class=10):
    
    for class_n in range(0,number_class,1):
        mean_l=[]
        var_l=[]
        for layer in range(angle_proj[class_n].shape[0]):
            m=statistics.mean(angle_proj[class_n][layer])
            mean_l.append(m)
            var_l.append(statistics.variance(angle_proj[class_n][layer], xbar =m))
        
        d = {'mean_layerwise':mean_l, 'var_layerwise':var_l}
        df1 = pd.DataFrame(data=d)
        df1.to_csv(f"{path}/mean_var/class{str(class_n)}{mean_var_name}_mean_var.csv")         
  