SET=$(seq -f "%02g" 1 21)
orginal="resnet50_csv_"
folder="unclass_PN_res50_pretrain"
convert_model="infer_unclass_res50_ep"

for i in $SET
do
        echo "--------------------------------------------------\n"
        echo "./keras-retinanet/keras_retinanet/bin/snapshots/$orginal"$i".h5"
        echo "running seq "$i
        python ./keras-retinanet/keras_retinanet/bin/convert_model.py ./keras-retinanet/keras_retinanet/bin/snapshots/$folder/$orginal$i.h5 ./keras-retinanet/keras_retinanet/bin/snapshots/$folder/$convert_model$i.h5
done
