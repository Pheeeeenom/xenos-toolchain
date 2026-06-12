.SUFFIXES:
ifeq ($(strip $(DEVKITXENON)),)
$(error "Please set DEVKITXENON in your environment. export DEVKITXENON=<path to>devkitPPC")
endif

include $(DEVKITXENON)/rules

TARGET		:=	$(notdir $(CURDIR))
BUILD		:=	build
SOURCES		:=	source
DATA		:=
INCLUDES	:=

CFLAGS	= -g -O2 -Wall $(MACHDEP) $(INCLUDE)
CXXFLAGS	=	$(CFLAGS)

LDFLAGS	=	-g $(MACHDEP) -Wl,--gc-sections -Wl,-Map,$(notdir $@).map

LIBS	:=	-lxenon -lm
LIBDIRS	:=

ifneq ($(BUILD),$(notdir $(CURDIR)))

export OUTPUT	:=	$(CURDIR)/$(TARGET)
export VPATH	:=	$(foreach dir,$(SOURCES),$(CURDIR)/$(dir)) \
						$(foreach dir,$(DATA),$(CURDIR)/$(dir))
export DEPSDIR	:=	$(CURDIR)/$(BUILD)

CFILES		:=	$(foreach dir,$(SOURCES),$(notdir $(wildcard $(dir)/*.c)))
CPPFILES	:=	$(foreach dir,$(SOURCES),$(notdir $(wildcard $(dir)/*.cpp)))
sFILES		:=	$(foreach dir,$(SOURCES),$(notdir $(wildcard $(dir)/*.s)))
SFILES		:=	$(foreach dir,$(SOURCES),$(notdir $(wildcard $(dir)/*.S)))
BINFILES	:=	$(foreach dir,$(DATA),$(notdir $(wildcard $(dir)/*.*)))

ifeq ($(strip $(CPPFILES)),)
	export LD	:=	$(CC)
else
	export LD	:=	$(CXX)
endif

export OFILES	:=	$(addsuffix .o,$(BINFILES)) \
						$(CPPFILES:.cpp=.o) $(CFILES:.c=.o) \
						$(sFILES:.s=.o) $(SFILES:.S=.o)

export INCLUDE	:=	$(foreach dir,$(INCLUDES),-I$(CURDIR)/$(dir)) \
						$(foreach dir,$(LIBDIRS),-I$(dir)/include) \
						-I$(CURDIR)/$(BUILD) \
						-I$(LIBXENON_INC)

export LIBPATHS	:=	$(foreach dir,$(LIBDIRS),-L$(dir)/lib) \
						-L$(LIBXENON_LIB)

export OUTPUT	:=	$(CURDIR)/$(TARGET)
.PHONY: $(BUILD) clean

$(BUILD):
	@[ -d $@ ] || mkdir -p $@
	@make --no-print-directory -C $(BUILD) -f $(CURDIR)/Makefile

clean:
	@echo clean ...
	@rm -fr $(BUILD) $(OUTPUT).elf $(OUTPUT).elf32

else

DEPENDS	:=	$(OFILES:.o=.d)

$(OUTPUT).elf32: $(OUTPUT).elf
$(OUTPUT).elf: $(OFILES)

-include $(DEPENDS)

endif
