SET=$(seq -f "%02g" 1 10)
orginal="snapshots/resnet101/b6_s2500/resnet101_csv_"
folder="snapshots/resnet101/b6_s2500/infer/"
convert_model="infer_resnet101_"

for i in $SET
do
	echo "--------------------------------------------------\n"
	echo "./keras-retinanet/keras_retinanet/bin/snapshots/"$orginal$i".h5"
	echo "running seq "$i
	python ./keras-retinanet/keras_retinanet/bin/convert_model.py ./keras-retinanet/keras_retinanet/bin/snapshots/$orginal$i.h5 ./keras-retinanet/keras_retinanet/bin/snapshots/$folder/$convert_model$i.h5
done
