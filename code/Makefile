SRCDIR=src
default: all

all:
	(cd $(SRCDIR) && $(MAKE))
	mv src/./tarea02 ./tarea02

clean:
	(cd $(SRCDIR) && $(MAKE) clean)
	mv src/./tarea02 ./tarea02

distclean:
	(cd $(SRCDIR) && $(MAKE) clean && $(MAKE) distclean)
	mv src/./tarea02 ./tarea02


.PHONY: all clean distclean