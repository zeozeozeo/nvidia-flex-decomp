// Headless Ghidra decompiler dump for flexdecomp.
// usage: analyzeHeadless <proj> flex -import <obj> -scriptPath tools/ghidra \
//          -postScript FlexDecomp.java <output.c> [-deleteProject]
import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;

import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileResults;
import ghidra.app.script.GhidraScript;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionManager;

public class FlexDecomp extends GhidraScript {
    @Override
    public void run() throws Exception {
        String[] args = getScriptArgs();
        if (args.length < 1) {
            printerr("FlexDecomp: output path required");
            return;
        }
        File outFile = new File(args[0]);
        int timeout = args.length > 1 ? Integer.parseInt(args[1]) : 180;

        DecompInterface ifc = new DecompInterface();
        ifc.toggleCCode(true);
        ifc.toggleSyntaxTree(true);
        if (!ifc.openProgram(currentProgram)) {
            printerr("FlexDecomp: cannot open program");
            return;
        }

        StringBuilder sb = new StringBuilder();
        sb.append("// decompiled by Ghidra ").append(getGhidraVersion())
          .append(" for flexdecomp\n// program: ").append(currentProgram.getName()).append("\n\n");

        FunctionManager fm = currentProgram.getFunctionManager();
        int done = 0, failed = 0;
        for (Function f : fm.getFunctions(true)) {
            if (f.isThunk() || f.isExternal()) continue;
            if (monitor.isCancelled()) break;
            sb.append("// ---- ").append(f.getName()).append("  @ ").append(f.getEntryPoint())
              .append("  (").append(f.getBody().getNumAddresses()).append(" bytes)\n");
            DecompileResults res = ifc.decompileFunction(f, timeout, monitor);
            if (res != null && res.decompileCompleted() && res.getDecompiledFunction() != null) {
                sb.append(res.getDecompiledFunction().getC());
                done++;
            } else {
                sb.append("// decompile failed: ")
                  .append(res == null ? "null result" : res.getErrorMessage()).append("\n");
                failed++;
            }
            sb.append("\n");
        }
        Files.write(Paths.get(outFile.getAbsolutePath()), sb.toString().getBytes(StandardCharsets.UTF_8));
        println("FlexDecomp: " + done + " functions, " + failed + " failures -> " + outFile);
        ifc.dispose();
    }
}
