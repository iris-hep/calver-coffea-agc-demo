from __future__ import absolute_import

import HiggsAnalysis.CombinedLimit.util.plotting as plot
import ROOT
import argparse
import ctypes

ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.gROOT.SetBatch(ROOT.kTRUE)

plot.ModTDRStyle()

parser = argparse.ArgumentParser(description="Arguments parser")
parser.add_argument("--input_file", dest="input_file", help="Input ROOT file from fitDiagnostics", required=True)
parser.add_argument(
    "--shape_type", dest="shape_type", help="Shape directory in input ROOT file from fitDiagnostics (e.g. shapes_fit_b, shapes_fit_s)", required=True
)
parser.add_argument("--region", dest="cards_dir", help="Region of interest (e.g. signal_region, ch1, ch2, etc.)", required=True)
parser.add_argument("--extra_suffix", dest="extra_suffix", required=False, default="")
args = parser.parse_args()

canvas = ROOT.TCanvas()

fin = ROOT.TFile(args.input_file)

first_dir = args.shape_type
second_dir = args.cards_dir

# Retrieve histograms
h_bkg = fin.Get(first_dir + "/" + second_dir + "/total_background")
h_sig = fin.Get(first_dir + "/" + second_dir + "/total_signal")
h_dat = fin.Get(first_dir + "/" + second_dir + "/data")  # TGraphAsymmErrors

# Set histogram styles
h_bkg.SetFillColor(ROOT.TColor.GetColor(100, 192, 232))  # Light blue background
h_sig.SetFillColor(ROOT.kRed)
h_sig.SetLineColor(ROOT.kRed)

# Create stack
hs = ROOT.THStack("hs", "")
hs.Add(h_bkg)
hs.Add(h_sig)

# Compute total uncertainty band (signal + background)
h_total = h_bkg.Clone("h_total")
h_total.Add(h_sig)  # Sum signal and background histograms

h_err = h_total.Clone("h_err")
for i in range(1, h_err.GetNbinsX() + 1):
    # add small quantity to fix problem with ROOT plotting - shade is shown anyways if error is 0
    bkg_err = h_bkg.GetBinError(i) + 10**-3
    #print(bkg_err)
    sig_err = h_sig.GetBinError(i) + 10**-3
    #print(sig_err)
    total_err = (bkg_err**2 + sig_err**2) ** 0.5  # Quadrature sum
    #print(total_err)
    h_err.SetBinError(i, total_err)

h_err.SetFillColorAlpha(12, 0.3)  # Grey uncertainty band
h_err.SetMarkerSize(0)

# update the edges
h_bkg.GetXaxis().SetLimits(110, 550)
h_sig.GetXaxis().SetLimits(110, 550)
#hs.GetXaxis().SetLimits(110, 550)
h_total.GetXaxis().SetLimits(110, 550)
h_err.GetXaxis().SetLimits(110, 550)
h_dat.GetXaxis().SetLimits(110, 550)

# Determine the correct y-axis range
max_y = max(h_total.GetMaximum() + h_err.GetBinError(h_err.GetMaximumBin()), h_dat.GetHistogram().GetMaximum())
hs.SetMaximum(max_y * 1.2)
h_err.SetMaximum(max_y * 1.2)

# rebin data
bin_edges = [110 + i * 40 for i in range(12)]
x_vals = []
y_vals = []
x_errs_low = []
x_errs_high = []
y_errs_low = []
y_errs_high = []
n_points = h_dat.GetN()

# Loop through each point in the graph
for i in range(n_points):
    x = ctypes.c_double(0)
    y = ctypes.c_double(0)
    h_dat.GetPoint(i, x, y)  # Get the i-th point

    # Calculate the bin center
    bin_index = i  # Assuming points are in the correct order for binning
    bin_center = (bin_edges[bin_index] + bin_edges[bin_index + 1]) / 2.0
    
    # Update the X value to be the bin center
    x_vals.append(bin_center)
    y_vals.append(y.value)

    # Set the X errors
    x_errs_low.append((bin_edges[bin_index + 1] - bin_edges[bin_index]) / 2)  # Half bin width
    x_errs_high.append((bin_edges[bin_index + 1] - bin_edges[bin_index]) / 2)  # Half bin width

    # Set the Y errors (assuming symmetrical errors, you can adjust as needed)
    y_errs_low.append(h_dat.GetErrorYlow(i))
    y_errs_high.append(h_dat.GetErrorYhigh(i))

# Create a new TGraphAsymmErrors with the updated points
new_graph = ROOT.TGraphAsymmErrors(len(x_vals))

# Set the new points and their errors
for i in range(len(x_vals)):
    new_graph.SetPoint(i, x_vals[i], y_vals[i])
    new_graph.SetPointError(i, x_errs_low[i], x_errs_high[i], y_errs_low[i], y_errs_high[i])

# Draw everything
hs.Draw("HIST")      # Stack plot
h_err.Draw("E2 SAME")  # Uncertainty band
#h_dat.Draw("P SAME")   # Data points
new_graph.Draw("P SAME")  # Data points

# Add legend
legend = ROOT.TLegend(0.60, 0.70, 0.90, 0.91, "", "NBNDC")
legend.AddEntry(h_bkg, "Background", "F")
legend.AddEntry(h_sig, "Signal", "F")
legend.AddEntry(h_err, "Total uncertainty", "F")
legend.Draw()

# Save the plot
canvas.SaveAs("combine_plots/stacked_plot_%s_%s%s.png" % (first_dir, second_dir, args.extra_suffix))
canvas.SaveAs("combine_plots/stacked_plot_%s_%s%s.pdf" % (first_dir, second_dir, args.extra_suffix))
