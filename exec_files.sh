# test_files.sh

python wec.py -c examples/structs_enums.wec
python wec.py -c examples/random.wec
python wec.py -c examples/inputs.wec
python wec.py -c examples/classes.wec
python wec.py -c examples/implements.wec
python wec.py -c examples/funcs_vars_etc.wec
python wec.py -c examples/arrays.wec
python wec.py -c examples/some_builtin_funcs.wec
python wec.py -c examples/files.wec
cd examples/import_test
python ../../wec.py -c main.wec
cd ../..
python wec.py -c examples/matrix.wec
python wec.py -c examples/simple_nn/main.wec
cd examples/pyimporttest
python ../../wec.py -c main.wec
cd ../..
cd examples/webtest
python ../../wec.py -c web.wec
cd ../..
python wec.py -c examples/loops.wec

echo "Done!"