#!/usr/bin/env python
# -*- coding: utf-8 -*-

# FIXME: Serialize extra vars to json file, and invoke using --extra-vars=@file
# FIXME: Fix all of the ansible templates that use 'None' as a string instead of null
# FIXME: Use openstacksdk to extract IPs


import argparse
import json
import os
import subprocess
import sys
import urllib
from urllib.parse import urlparse
from zipfile import ZipFile

spark_versions = {
    # 2.4.4 supports 3.1 as well, but handling SHA hashes in Ansible is hard
    "2.4.8": {"hadoop_versions": ["2.7"]},
    "2.3.0": {"hadoop_versions": ["2.6", "2.7"]},
    "2.2.1": {"hadoop_versions": ["2.6", "2.7"]},
    "2.2.0": {"hadoop_versions": ["2.6", "2.7"]},
    "2.1.0": {"hadoop_versions": ["2.3", "2.4", "2.6", "2.7"]},
    "2.0.2": {"hadoop_versions": ["2.3", "2.4", "2.6", "2.7"]},
    "2.0.1": {"hadoop_versions": ["2.3", "2.4", "2.6", "2.7"]},
    "2.0.0": {"hadoop_versions": ["2.3", "2.4", "2.6", "2.7"]},
    "1.6.2": {"hadoop_versions": ["1", "cdh4", "2.3", "2.4", "2.6"]},
    "1.6.1": {"hadoop_versions": ["1", "cdh4", "2.3", "2.4", "2.6"]},
    "1.6.0": {"hadoop_versions": ["1", "cdh4", "2.3", "2.4", "2.6"]},
    "1.5.2": {"hadoop_versions": ["1", "cdh4", "2.3", "2.4", "2.6"]},
    "1.5.1": {"hadoop_versions": ["1", "cdh4", "2.3", "2.4", "2.6"]},
    "1.5.0": {"hadoop_versions": ["1", "cdh4", "2.3", "2.4", "2.6"]},
    "1.4.1": {"hadoop_versions": ["1", "cdh4", "2.3", "2.4", "2.6"]},
    "1.4.0": {"hadoop_versions": ["1", "cdh4", "2.3", "2.4", "2.6"]},
    "1.3.1": {"hadoop_versions": ["1", "cdh4", "2.3", "2.4", "2.6"]},
    "1.3.0": {"hadoop_versions": ["1", "cdh4", "2.3", "2.4"]},
    "1.2.2": {"hadoop_versions": ["1", "cdh4", "2.3", "2.4"]},
    "1.2.1": {"hadoop_versions": ["1", "cdh4", "2.3", "2.4"]},
    "1.2.0": {"hadoop_versions": ["1", "cdh4", "2.3", "2.4"]},
    "1.1.1": {"hadoop_versions": ["1", "cdh4", "2.3", "2.4"]},
    "1.1.0": {"hadoop_versions": ["1", "cdh4", "2.3", "2.4"]},
    "1.0.2": {"hadoop_versions": ["1", "cdh4"]},
    "1.0.1": {"hadoop_versions": ["1", "cdh4"]},
    "1.0.0": {"hadoop_versions": ["1", "cdh4"]},
}

toree_versions = {
    "1": "0.1",
    "2": "0.4",
}


def abort(msg, status=1):
    print(msg, file=sys.stderr)
    sys.exit(status)


parser = argparse.ArgumentParser(
    description='Spark cluster deploy tools for Openstack.',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog='Usage real-life examples:\t\n'
    '   ./spark-openstack -k borisenko -i ~/.ssh/id_rsa -s 2 -t spark.large -a 20545e58-59de-4212-a83f-3703b31622cf -n computations-net -f external_network --async launch spark-cluster\n'
    '   ./spark-openstack --async destroy spark-cluster\n'
    'Look through README.md for more advanced usage examples.\n'
    'Apache 2.0, ISP RAS 2016 (http://ispras.ru/en).\n',
)

parser.add_argument(
    'act',
    type=str,
    choices=["launch", "destroy", "get-master", "config", "runner", "debug"],
)
parser.add_argument('cluster_name', help="Name for your cluster")
parser.add_argument('option', nargs='?')
parser.add_argument('--print-command', action='store_true')
parser.add_argument('-k', '--key-pair')
parser.add_argument("-i", "--identity-file")
parser.add_argument("-s", "--slaves", type=int)
parser.add_argument(
    "-n",
    "--virtual-network",
    help="Your virtual Openstack network id for cluster. If the cluster has only one network, you may not specify it",
)
parser.add_argument("-f", "--floating-ip-pool", help="Floating IP pool")
parser.add_argument("-t", "--instance-type")
parser.add_argument(
    "-m",
    "--master-instance-type",
    help="master instance type, defaults to slave instance type",
)
parser.add_argument("-a", "--image-id")
parser.add_argument("-w", help="ignored")

parser.add_argument(
    "--create", action="store_true", help="Note that cluster should be created"
)
parser.add_argument(
    "--deploy-spark",
    action="store_true",
    help="Request Spark deployment (using Hadoop)",
)
parser.add_argument(
    "--mountnfs",
    action="store_true",
    help="Request mountnfs",
)
parser.add_argument(
    "--spark-worker-mem-mb",
    type=int,
    help="Force worker memory value in megabytes (e.g. 14001)",
)
parser.add_argument(
    "-j",
    "--deploy-jupyter",
    action='store_true',
    help="Request Jupyter deployment on master node.",
)
parser.add_argument(
    "-jh",
    "--deploy-jupyterhub",
    action='store_true',
    help="Request JupyterHub deployment on master node",
)
parser.add_argument(
    "--spark-version", default="2.4.4", help="Spark version to use"
)
parser.add_argument("--hadoop-version", help="Hadoop version to use")
parser.add_argument(
    "--boot-from-volume",
    default=False,
    help="Boot cluster from Cinder volume.",
)
parser.add_argument(
    "--hadoop-user",
    default="ubuntu",
    help="User to use/create for cluster members",
)
parser.add_argument(
    "--ansible-bin",
    help="Path to ansible and ansible-playbook",
    default='',
)
parser.add_argument(
    "--swift-username",
    help="Username for Swift object storage. If not specified, swift integration "
    "is commented out in core-site.xml. You can also use OS_SWIFT_USERNAME"
    "environment variable",
)
parser.add_argument(
    "--swift-password",
    help="Username for Swift object storage. If not specified, swift integration "
    "is commented out in core-site.xml. You can also use OS_SWIFT_PASSWORD"
    "environment variable",
)
parser.add_argument(
    "--nfs-share",
    default=[],
    nargs=2,
    metavar=("<nfs-path>", "<mount-path>"),
    help="Mount NFS share(s) on instances",
    action='append',
)
parser.add_argument(
    "--extra-jars",
    action="append",
    help="Add/replace extra jars to Spark (during launch). Jar file names must be different",
)

parser.add_argument(
    "--yarn", action='store_true', help="Should we deploy using Apache YARN."
)

parser.add_argument(
    "--deploy-cassandra",
    action='store_true',
    help="Request Apache Cassandra deployment",
)
parser.add_argument(
    "--cassandra-version",
    default="3.11.4",
    help="Apache Cassandra version",
)
parser.add_argument(
    "--skip-packages",
    action='store_true',
    help="Skip package installation (Java, rsync, etc). Image must contain all required packages.",
)
parser.add_argument(
    "--skip-prepare",
    action='store_true',
    help="Skip certain preparation steps. Useful when rerunning after failure.",
)
parser.add_argument(
    "--async",
    action="store_true",
    dest='async_',
    help="Async Openstack operations (may not work with some Openstack environments)",
)
parser.add_argument("--tags", help="Ansible: run specified tags")
parser.add_argument("--skip-tags", help="Ansible: skip specified tags")


args, unknown = parser.parse_known_args()
if args.tags is not None:
    unknown.append("--tags")
    unknown.append(args.tags)

if args.skip_tags is not None:
    unknown.append("--skip-tags")
    unknown.append(args.skip_tags)

if args.master_instance_type is None:
    args.master_instance_type = args.instance_type

if "_" in args.cluster_name:
    abort("Underscores in cluster name are not supported")

ansible_cmd = "ansible"
ansible_playbook_cmd = "ansible-playbook"
if args.ansible_bin is not None:
    ansible_cmd = os.path.join(args.ansible_bin, "ansible")
    ansible_playbook_cmd = os.path.join(args.ansible_bin, "ansible-playbook")


def get_cassandra_connector_jar(spark_version):
    spark_cassandra_connector_url = (
        "http://dl.bintray.com/spark-packages/maven/datastax/spark-cassandra-connector/1.6.8-s_2.10/spark-cassandra-connector-1.6.8-s_2.10.jar"
        if args.spark_version.startswith("1.6")
        else "http://dl.bintray.com/spark-packages/maven/datastax/spark-cassandra-connector/2.0.3-s_2.11/spark-cassandra-connector-2.0.3-s_2.11.jar"
    )

    spark_cassandra_connector_filename = "/tmp/" + os.path.basename(
        urlparse.urlsplit(spark_cassandra_connector_url).path
    )

    if not os.path.exists(spark_cassandra_connector_filename):
        print(
            "Downloading Spark Cassandra Connector for Spark version {0}".format(
                spark_version
            )
        )
        urllib.urlretrieve(
            spark_cassandra_connector_url,
            filename=spark_cassandra_connector_filename,
        )

    return spark_cassandra_connector_filename


def make_extra_vars(action: str = args.act):
    extra_vars = {}
    mapping = dict(
        act='act',
        n_slaves='slaves',
        cluster_name='cluster_name',
        os_image='image_id',
        os_key_name='key_pair',
        flavor='instance_type',
        master_flavor='master_instance_type',
        virtual_network='virtual_network',
        ansible_user='hadoop_user',
        ansible_ssh_private_key_file='identity_file',
        hadoop_user='hadoop_user',
    )
    extra_vars.update(
        (target, getattr(args, key)) for target, key in mapping.items()
    )
    # HACK: There's a lot of Ansible code that checks against the string 'None'
    extra_vars['floating_ip_pool'] = args.floating_ip_pool or 'None'

    extra_vars["os_project_name"] = os.getenv('OS_PROJECT_NAME') or os.getenv(
        'OS_TENANT_NAME'
    )
    extra_vars["os_auth_url"] = os.getenv('OS_AUTH_URL')
    if not extra_vars["os_project_name"] or not extra_vars["os_auth_url"]:
        abort("Please source your OpenStack openrc file", -1)

    if action == 'launch':
        extra_vars["create_cluster"] = args.create
        extra_vars["deploy_spark"] = args.deploy_spark
        extra_vars["mountnfs"] = args.mountnfs
        extra_vars["spark_version"] = args.spark_version
        if args.hadoop_version:
            if (
                args.hadoop_version
                not in spark_versions[args.spark_version]["hadoop_versions"]
            ):
                abort(
                    "The chosen Spark version doesn't support the selected Hadoop version!",
                    -1,
                )
            extra_vars["hadoop_version"] = args.hadoop_version
        else:
            extra_vars["hadoop_version"] = spark_versions[args.spark_version][
                "hadoop_versions"
            ][-1]
        print(
            "Deploying Apache Spark %s with Apache Hadoop %s"
            % (extra_vars["spark_version"], extra_vars["hadoop_version"])
        )

    extra_vars["boot_from_volume"] = args.boot_from_volume

    extra_vars["os_swift_username"] = (
        args.swift_username or os.getenv('OS_SWIFT_USERNAME') or None
    )
    if not extra_vars["os_swift_username"]:
        del extra_vars["os_swift_username"]
    extra_vars["os_swift_password"] = (
        args.swift_password or os.getenv('OS_SWIFT_PASSWORD') or None
    )
    if not extra_vars["os_swift_password"]:
        del extra_vars["os_swift_password"]

    extra_vars["deploy_jupyter"] = args.deploy_jupyter
    if args.deploy_jupyter:
        extra_vars['toree_version'] = toree_versions[
            extra_vars['spark_version'][0]
        ]

    extra_vars["deploy_jupyterhub"] = args.deploy_jupyterhub
    extra_vars["nfs_shares"] = [
        {"nfs_path": nfs, "mount_path": mount} for nfs, mount in args.nfs_share
    ]

    extra_vars["use_yarn"] = args.yarn

    # Cassandra deployment => --extra-args
    extra_vars["deploy_cassandra"] = args.deploy_cassandra
    extra_vars["cassandra_version"] = args.cassandra_version

    extra_vars["skip_packages"] = args.skip_packages

    extra_vars["sync"] = "async" if args.async_ else "sync"

    if args.extra_jars is None:
        args.extra_jars = []

    extra_jars = []

    def add_jar(path):
        extra_jars.append(
            {"name": os.path.basename(path), "path": os.path.abspath(path)}
        )

    for jar in args.extra_jars:
        if os.path.isdir(jar):
            for f in os.listdir(jar):
                add_jar(os.path.join(jar, f))
        else:
            add_jar(jar)

    # Obtain Cassandra connector jar if cassandra is deployed
    if args.deploy_cassandra:
        cassandra_jar = get_cassandra_connector_jar(args.spark_version)
        add_jar(cassandra_jar)

    extra_vars["extra_jars"] = extra_jars

    return extra_vars


def parse_host_ip(resp):
    """parse ansible debug output with var=hostvars[inventory_hostname].ansible_ssh_host and return host"""
    parts1 = resp.split("=>")
    if len(parts1) != 2:
        abort("unexpected ansible output")
    parts2 = parts1[1].split(":")
    if len(parts2) != 3:
        abort("unexpected ansible output")
    parts3 = parts2[1].split('"')
    if len(parts3) != 3:
        abort("unexpected ansible output")
    return parts3[1]


def get_master_ip():
    extra_vars = make_extra_vars()
    extra_vars['extended_role'] = 'master'
    res = subprocess.check_output(
        [
            ansible_playbook_cmd,
            "--extra-vars",
            json.dumps(extra_vars),
            "get_ip.yml",
        ]
    )
    return parse_host_ip(res)


def get_ip(role):
    extra_vars = make_extra_vars()
    extra_vars['extended_role'] = role
    res = subprocess.check_output(
        [
            ansible_playbook_cmd,
            "--extra-vars",
            json.dumps(extra_vars),
            "get_ip.yml",
        ]
    )
    return parse_host_ip(res)


cmdline = [ansible_playbook_cmd]
cmdline.extend(unknown)

extra_vars = make_extra_vars()

if args.act == "launch":
    cmdline_create = cmdline[:]
    cmdline_create.extend(["main.yml", "--extra-vars", json.dumps(extra_vars)])
    if args.skip_prepare:
        cmdline_create.extend(['--skip-tags', 'prepare'])
    if args.print_command:
        cmdline_create[-1] = '"%s"' % json.dumps(extra_vars)
        print(' '.join(cmdline_create))
        sys.exit(0)
    subprocess.call(cmdline_create)
    # master_ip = get_master_ip()
    # print("Cluster launched successfully; Master IP is %s" % (master_ip))
elif args.act == "destroy":
    res = subprocess.check_output(
        [
            ansible_cmd,
            "--extra-vars",
            repr(make_extra_vars()),
            "-m",
            "debug",
            "-a",
            "var=groups['%s_slaves']" % args.cluster_name,
            args.cluster_name + "-master",
        ]
    )
    extra_vars = make_extra_vars()
    cmdline_create = cmdline[:]
    cmdline_create.extend(["main.yml", "--extra-vars", json.dumps(extra_vars)])
    subprocess.call(cmdline_create)
elif args.act == "get-master":
    print(get_master_ip())
elif args.act == "config":
    extra_vars = make_extra_vars()
    extra_vars['roles_dir'] = '../roles'

    cmdline_inventory = cmdline[:]
    if (
        args.option == 'restart-spark'
    ):  # Skip installation tasks, run only detect_conf tasks
        cmdline_inventory.extend(("--skip-tags", "spark_install"))

    elif args.option == 'restart-cassandra':
        cmdline_inventory.extend(("--skip-tags", "spark_install,cassandra"))

    cmdline_inventory.extend(
        ["%s.yml" % args.option, "--extra-vars", json.dumps(extra_vars)]
    )
    subprocess.call(cmdline_inventory)
elif args.act == "runner":
    cmdline_create = cmdline[:]
    cmdline_create.extend(
        ["prepare_internal_runner.yml", "--extra-vars", json.dumps(extra_vars)]
    )
    subprocess.call(cmdline_create)
    runner_ip = get_ip('runner')
    print("Runner ready; IP is %s" % (runner_ip))
elif args.act == 'debug':
    print(extra_vars)
