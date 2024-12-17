if [ $1 ] 
then
    scp -i ~/.ssh/id_ed25519 -r anton@glaurung.pc.sdu.dk:~/fusion/investigations/scatem/scanem_list/$1 ./data/CTS
else
    scp -i ~/.ssh/id_ed25519 -r anton@glaurung.pc.sdu.dk:~/fusion/investigations/scatem/scanem_list/ ./data/CTS
fi