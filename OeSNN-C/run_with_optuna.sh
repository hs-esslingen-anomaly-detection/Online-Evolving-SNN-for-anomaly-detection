
if [ -f ../config.sh ]; then
    chmod +x ../config.sh
    source ../config.sh
else
    TRIALS_PER_CORE=10
    CPU_CORES=24
fi

echo "Build OeSNN"
make clean build
echo "Stat hparam search with $CPU_CORES CPU cores and $TRIALS_PER_CORE trials per core"
echo "init ..."
docker stop optuna_mysql >/dev/null 2>&1
docker rm optuna_mysql >/dev/null 2>&1
sleep 2
echo "start mysql database ..."
docker run --rm -p 3306:3306 --name optuna_mysql -e MYSQL_ROOT_PASSWORD=Geheim -e MYSQL_DATABASE=optuna -d mysql
sleep 16
echo "create database ..."
optuna create-study --study-name "distributed" --direction='minimize' --storage "mysql://root:Geheim@127.0.0.1/optuna" --skip-if-exists
sleep 1

close_hook() {
    echo "exit"
    pkill python
    exit
}
trap close_hook SIGHUP SIGINT SIGTERM EXIT

echo "start hparam search ..."

for i in $(seq 1 $CPU_CORES)
do
    python ./optuna_mysql_train.py $TRIALS_PER_CORE &
    PID=$!
done
wait $PID
sleep 8
echo "start validation ..."
python ./optuna_mysql_test.py
