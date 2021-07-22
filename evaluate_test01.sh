SET=$(seq -f "%02g" 13 20)
orginal="resnet50_csv_"
folder="unclass_PN_train"
evaluate_model="infer_unclass_res50_ep"

for i in $SET
do
        echo "--------------------------------------------------\n"
        echo "./keras-retinanet/keras_retinanet/bin/snapshots/$orginal"$i".h5"
        echo "running seq "$i
        python ./keras-retinanet/keras_retinanet/bin/evaluate.py csv ./meta_ver2_test.csv ./class/meta/mappingclass.csv ./keras-retinanet/keras_retinanet/bin/snapshots/$folder/$evaluate_model$i.h5
done
