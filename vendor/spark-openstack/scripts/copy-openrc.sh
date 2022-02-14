#! /bin/bash

ssh -T jump.cloudlab.internal -- \
	sudo cat /root/setup/admin-openrc.sh \
> admin-openrc.sh
# ssh -T mgt-node.cloudlab.internal -- \
# 	tee spark-openstack/admin-openrc.sh \
# < admin-openrc.sh
