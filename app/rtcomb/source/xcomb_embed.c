/* xcomb_embed.c — builds the runtime combiner generator into the app.
 *
 * The devkitxenon rules file only compiles *.c found via the Makefile's
 * SOURCES dirs (through VPATH), so rather than adding a second source dir
 * or copying files around at build time, this thin translation unit pulls
 * in the canonical xcomb/xcomb.c by relative include.  Quoted includes
 * resolve relative to THIS file, so "../../../xcomb/xcomb.c" lands on
 * <toolchain root>/xcomb/xcomb.c, and xcomb.c's own includes of xcomb.h /
 * xcomb_templates.h resolve next to it.  No copies, single source of
 * truth, byte-identical to what the host tests exercise.
 */
#include "../../../xcomb/xcomb.c"
