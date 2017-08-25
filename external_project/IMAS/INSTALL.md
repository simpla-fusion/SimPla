#Pre-Require
python-numpy cython libsaxonhe-java

#ENV
```
export CLASSPATH=/usr/share/java/saxon9he.jar
export PKG_CONFIG_PATH=/usr/lib/pkgconfig/
```
#Build
```
WORK_DIR=/work/imas

cd $WORK_DIR/

git clone ssh://git@git.iter.org/imas/installer.git
git clone ssh://git@git.iter.org/imex/kepler-installer.git


module load Blitz++/0.10 MDSplus/6.1.84 Python/2.7 Java/1.8  intel/2016

export IMAS_HOME=/pkg/imas
export UAL_VERSION=3.4.0
export IMAS_VERSION=3.7.4

cd $WORK_DIR/installer

make update
make version
make dd
MATLAB=no PGI=no make ual -j8
make install
make module_install 

cd $WORK_DIR/kepler-installer

 ./install-kepler-2.5-bare.sh /pkg/imas/extra/kepler/2.5-imas-3.7.4

export KEPLER=/pkg/imas/extra/kepler/2.5-imas-3.7.4

./install-kepler-modules.sh /pkg/imas/extra/ /pkg/imas/etc/modulefiles/
```
# Run
```
module load ant/1.9.6 
module load kepler/2.5-imas-3.7.4 

cd $IMAS_HOME/extra/
git clone  ssh://git@git.iter.org/imex/fc2k.git
cd $IMAS_HOME/extra/fc2k
ant jar
```