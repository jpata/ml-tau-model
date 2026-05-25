import os
import json
import matplotlib
import numpy as np
import awkward as ak
import mplhep as hep
from matplotlib import ticker
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from mltau.tools import general as g
from mltau.tools.evaluation.histogram import Histogram
from mltau.tools.io.general import NpEncoder

hep.style.use(hep.styles.CMS)
matplotlib.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "stix"


class TaggerEvaluator:
    def __init__(
        self,
        signal_predictions: np.array,
        signal_gen_tau_p4: ak.Array,
        signal_reco_jet_p4: ak.Array,
        bkg_predictions: np.array,
        bkg_gen_jet_p4: ak.Array,
        bkg_reco_jet_p4: ak.Array,
        cfg: DictConfig,
        sample: str,
        algorithm: str,
    ):
        self.signal_predictions = signal_predictions
        self.signal_gen_tau_p4 = g.reinitialize_p4(signal_gen_tau_p4)
        self.signal_reco_jet_p4 = g.reinitialize_p4(signal_reco_jet_p4)

        self.bkg_predictions = bkg_predictions
        self.bkg_gen_jet_p4 = g.reinitialize_p4(bkg_gen_jet_p4)
        self.bkg_reco_jet_p4 = g.reinitialize_p4(bkg_reco_jet_p4)
        self.cfg = cfg
        self.sample = sample  # TODO: Actually not used anymore
        self.algorithm = algorithm
        # Use quantile-based thresholds: dense where scores concentrate (near 0/1),
        # sparse in the middle. Capped at 1000 points for performance.
        all_scores = np.concatenate(
            [np.asarray(signal_predictions), np.asarray(bkg_predictions)]
        )
        self.tagging_cuts = np.unique(
            np.concatenate([[0], np.quantile(all_scores, np.linspace(0, 1, 1000)), [1]])
        )

        self.fakerates, self.fake_numerator_mask, self.fake_denominator_mask = (
            self._calculate_fakerates()
        )
        self.efficiencies, self.eff_numerator_mask, self.eff_denominator_mask = (
            self._calculate_efficiencies()
        )

        self.loose_wp, self.medium_wp, self.tight_wp = self._calculate_wps()
        self.wp_metrics = {}
        for name, metric in cfg.metrics.tagging.metrics.items():
            fr_bin_centers, fr_data, fr_yerr, fr_xerr = (
                self._get_working_point_fakerates(name, metric)
            )
            eff_bin_centers, eff_data, eff_yerr, eff_xerr = (
                self._get_working_point_efficiencies(name, metric)
            )
            self.wp_metrics[name] = {
                "fakerates": fr_data,
                "fr_bin_centers": fr_bin_centers,
                "fr_yerr": fr_yerr,
                "fr_xerr": fr_xerr,
                "efficiencies": eff_data,
                "eff_bin_centers": eff_bin_centers,
                "eff_yerr": eff_yerr,
                "eff_xerr": eff_xerr,
            }

    def _calculate_fakerates(self):
        fakerates = []
        # Denominator
        ref_var_pt_mask = self.bkg_gen_jet_p4.pt > self.cfg.metrics.tagging.cuts.min_pt
        ref_var_theta_mask1 = (
            abs(np.rad2deg(self.bkg_gen_jet_p4.theta))
            < self.cfg.metrics.tagging.cuts.max_theta
        )
        ref_var_theta_mask2 = (
            abs(np.rad2deg(self.bkg_gen_jet_p4.theta))
            > self.cfg.metrics.tagging.cuts.min_theta
        )
        denominator_mask = ref_var_pt_mask * ref_var_theta_mask1 * ref_var_theta_mask2

        # Numerator
        tau_pt_mask = self.bkg_reco_jet_p4.pt > self.cfg.metrics.tagging.cuts.min_pt
        tau_theta_mask1 = (
            abs(np.rad2deg(self.bkg_reco_jet_p4.theta))
            < self.cfg.metrics.tagging.cuts.max_theta
        )
        tau_theta_mask2 = (
            abs(np.rad2deg(self.bkg_reco_jet_p4.theta))
            > self.cfg.metrics.tagging.cuts.min_theta
        )
        numerator_mask = tau_pt_mask * tau_theta_mask1 * tau_theta_mask2
        numerator_mask = numerator_mask * denominator_mask
        n_all = np.sum(denominator_mask)
        for cut in self.tagging_cuts:
            n_passing_cuts = np.sum(self.bkg_predictions[numerator_mask] > cut)
            fakerate = n_passing_cuts / n_all
            fakerates.append(fakerate)
        return fakerates, numerator_mask, denominator_mask

    def _calculate_efficiencies(self):
        efficiencies = []
        # Denominator
        ref_var_pt_mask = (
            self.signal_gen_tau_p4.pt > self.cfg.metrics.tagging.cuts.min_pt
        )
        ref_var_theta_mask1 = (
            abs(np.rad2deg(self.signal_gen_tau_p4.theta))
            < self.cfg.metrics.tagging.cuts.max_theta
        )
        ref_var_theta_mask2 = (
            abs(np.rad2deg(self.signal_gen_tau_p4.theta))
            > self.cfg.metrics.tagging.cuts.min_theta
        )
        denominator_mask = ref_var_pt_mask * ref_var_theta_mask1 * ref_var_theta_mask2

        # Numerator
        tau_pt_mask = self.signal_reco_jet_p4.pt > self.cfg.metrics.tagging.cuts.min_pt
        tau_theta_mask1 = (
            abs(np.rad2deg(self.signal_reco_jet_p4.theta))
            < self.cfg.metrics.tagging.cuts.max_theta
        )
        tau_theta_mask2 = (
            abs(np.rad2deg(self.signal_reco_jet_p4.theta))
            > self.cfg.metrics.tagging.cuts.min_theta
        )
        numerator_mask = tau_pt_mask * tau_theta_mask1 * tau_theta_mask2
        numerator_mask = numerator_mask * denominator_mask

        n_all = np.sum(denominator_mask)
        for cut in self.tagging_cuts:
            n_passing_cuts = np.sum(self.signal_predictions[numerator_mask] > cut)
            efficiency = n_passing_cuts / n_all
            efficiencies.append(efficiency)
        return efficiencies, numerator_mask, denominator_mask

    def _calculate_wps(self):
        working_points = {"Loose": 0.7, "Medium": 0.8, "Tight": 0.9}  # Efficiencies
        wp_values = {}
        for wp_name, wp_value in working_points.items():
            diff = abs(np.array(self.efficiencies) - wp_value)
            idx = np.argmin(diff)
            if not diff[idx] > 0.1:
                cut = self.tagging_cuts[idx]
            else:
                cut = -1
            wp_values[wp_name] = cut
        return wp_values["Loose"], wp_values["Medium"], wp_values["Tight"]

    def _get_working_point_efficiencies(self, name, metric):
        medium_wp_mask = self.signal_predictions > self.medium_wp
        var_values = getattr(self.signal_gen_tau_p4, name).to_numpy()
        if name == "theta":
            var_values = np.rad2deg(var_values)
        eff_var_denom = var_values[self.eff_denominator_mask]
        eff_var_num = var_values[medium_wp_mask * self.eff_numerator_mask]
        
        if len(eff_var_denom) == 0:
            # Return dummy values if empty to avoid crash in debug runs
            n_bins = metric.n_bins
            return (
                np.zeros(n_bins),
                np.zeros(n_bins),
                np.zeros(n_bins),
                np.zeros(n_bins),
            )

        bin_edges = np.linspace(
            min(eff_var_denom), max(eff_var_denom), metric.n_bins + 1
        )
        denom_hist = Histogram(eff_var_denom, bin_edges, "denominator")
        num_hist = Histogram(eff_var_num, bin_edges, "numerator")
        efficiencies = num_hist / denom_hist
        return (
            efficiencies.bin_centers,
            efficiencies.data,
            efficiencies.uncertainties,
            efficiencies.bin_halfwidths,
        )

    def _get_working_point_fakerates(self, name, metric):
        medium_wp_mask = self.bkg_predictions > self.medium_wp
        var_values = getattr(self.bkg_gen_jet_p4, name).to_numpy()
        if name == "theta":
            var_values = np.rad2deg(var_values)
        fake_var_denom = var_values[self.fake_denominator_mask]
        fake_var_num = var_values[medium_wp_mask * self.fake_numerator_mask]
        
        if len(fake_var_denom) == 0:
            # Return dummy values if empty to avoid crash in debug runs
            n_bins = metric.n_bins
            return (
                np.zeros(n_bins),
                np.zeros(n_bins),
                np.zeros(n_bins),
                np.zeros(n_bins),
            )

        bin_edges = np.linspace(
            min(fake_var_denom), max(fake_var_denom), metric.n_bins + 1
        )
        denom_hist = Histogram(fake_var_denom, bin_edges, "denominator")
        num_hist = Histogram(fake_var_num, bin_edges, "numerator")
        fakerates = num_hist / denom_hist
        return (
            fakerates.bin_centers,
            fakerates.data,
            fakerates.uncertainties,
            fakerates.bin_halfwidths,
        )


class ROCPlot:
    def __init__(self, cfg):
        self.fig, self.ax = self.plot()
        self.cfg = cfg

    def plot(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_ylabel(r"$P_{misid}$", fontsize=30)
        ax.set_xlabel(r"$\varepsilon_{\tau}$", fontsize=30)
        ax.tick_params(axis="x", labelsize=30)
        ax.tick_params(axis="y", labelsize=30)
        ax.set_ylim((1e-5, 1))
        ax.set_xlim((0, 1))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.set_yscale("log")
        plt.grid()
        return fig, ax

    def add_line(self, evaluator):
        self.ax.plot(
            evaluator.efficiencies,
            evaluator.fakerates,
            color=self.cfg.metrics.ALGORITHM_PLOT_STYLES[evaluator.algorithm].color,
            marker=self.cfg.metrics.ALGORITHM_PLOT_STYLES[evaluator.algorithm].marker,
            label=self.cfg.metrics.ALGORITHM_PLOT_STYLES[evaluator.algorithm].name,
            ms=15,
            ls="",
        )
        self.ax.legend(prop={"size": 30})

    def save(self, output_path):
        self.fig.savefig(output_path, format="pdf")
        plt.close("all")


class FakeRatePlot:
    def __init__(self, cfg: DictConfig, metric: str):
        self.cfg = cfg
        self.metric = metric
        self.fig, self.ax = self.plot()

    def add_line(self, evaluator):
        self.ax.errorbar(
            evaluator.wp_metrics[self.metric]["fr_bin_centers"],
            evaluator.wp_metrics[self.metric]["fakerates"],
            xerr=evaluator.wp_metrics[self.metric]["fr_xerr"],
            yerr=evaluator.wp_metrics[self.metric]["fr_yerr"],
            ms=20,
            color=self.cfg.metrics.ALGORITHM_PLOT_STYLES[evaluator.algorithm].color,
            marker=self.cfg.metrics.ALGORITHM_PLOT_STYLES[evaluator.algorithm].marker,
            linestyle="",
            label=self.cfg.metrics.ALGORITHM_PLOT_STYLES[evaluator.algorithm].name,
        )
        self.ax.legend(loc="upper right", prop={"size": 30})

    def plot(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.xaxis.set_major_locator(
            ticker.MultipleLocator(
                self.cfg.metrics.tagging.metrics[self.metric].x_maj_tick_spacing
            )
        )
        ax.set_xlabel(
            rf"{self.cfg.metrics.tagging.performances.fakerate.xlabel[self.metric]}",
            fontsize=30,
        )
        ax.set_ylabel(
            rf"{self.cfg.metrics.tagging.performances.fakerate.ylabel}", fontsize=30
        )
        ax.set_yscale(self.cfg.metrics.tagging.performances.fakerate.yscale)
        if self.cfg.metrics.tagging.performances.fakerate.ylim is not None:
            ylim = tuple(self.cfg.metrics.tagging.performances.fakerate.ylim)
        else:
            ylim = self.cfg.metrics.tagging.performances.fakerate.ylim
        ax.set_ylim(ylim)
        ax.tick_params(axis="x", labelsize=30)
        ax.tick_params(axis="y", labelsize=30)
        plt.grid()
        return fig, ax

    def save(self, output_path):
        self.fig.savefig(output_path, format="pdf")
        plt.close("all")


class EfficiencyPlot:
    def __init__(self, cfg: DictConfig, metric: str):
        self.cfg = cfg
        self.metric = metric
        self.fig, self.ax = self.plot()

    def add_line(self, evaluator):
        self.ax.errorbar(
            evaluator.wp_metrics[self.metric]["eff_bin_centers"],
            evaluator.wp_metrics[self.metric]["efficiencies"],
            xerr=evaluator.wp_metrics[self.metric]["eff_xerr"],
            yerr=evaluator.wp_metrics[self.metric]["eff_yerr"],
            ms=20,
            color=self.cfg.metrics.ALGORITHM_PLOT_STYLES[evaluator.algorithm].color,
            marker=self.cfg.metrics.ALGORITHM_PLOT_STYLES[evaluator.algorithm].marker,
            linestyle="",
            label=self.cfg.metrics.ALGORITHM_PLOT_STYLES[evaluator.algorithm].name,
        )
        self.ax.legend(loc="upper right", prop={"size": 30})

    def plot(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.xaxis.set_major_locator(
            ticker.MultipleLocator(
                self.cfg.metrics.tagging.metrics[self.metric].x_maj_tick_spacing
            )
        )
        ax.set_xlabel(
            rf"{self.cfg.metrics.tagging.performances.efficiency.xlabel[self.metric]}",
            fontsize=30,
        )
        ax.set_ylabel(
            rf"{self.cfg.metrics.tagging.performances.efficiency.ylabel}",
            fontsize=30,
        )
        ax.set_yscale(self.cfg.metrics.tagging.performances.efficiency.yscale)
        if self.cfg.metrics.tagging.performances.efficiency.ylim is not None:
            ylim = tuple(self.cfg.metrics.tagging.performances.efficiency.ylim)
        else:
            ylim = self.cfg.metrics.tagging.performances.efficiency.ylim
        ax.set_ylim(tuple(ylim))
        ax.tick_params(axis="x", labelsize=30)
        ax.tick_params(axis="y", labelsize=30)
        plt.grid()
        return fig, ax

    def save(self, output_path):
        self.fig.savefig(output_path, format="pdf")
        plt.close("all")


class TauClassifierPlot:
    def __init__(self):
        self.bin_edges = np.linspace(start=0, stop=1, num=21)
        self.fig, self.ax = self.plot()

    def add_line(self, evaluator, dataset: str):
        linestyle = "solid" if dataset == "test" else "dashed"
        bkg_histogram = np.histogram(evaluator.bkg_predictions, bins=self.bin_edges)[0]
        bkg_histogram = bkg_histogram / np.sum(bkg_histogram)
        signal_histogram = np.histogram(
            evaluator.signal_predictions, bins=self.bin_edges
        )[0]
        signal_histogram = signal_histogram / np.sum(signal_histogram)
        hep.histplot(
            signal_histogram,
            bins=self.bin_edges,
            histtype="step",
            label="Signal",
            ls=linestyle,
            color="red",
            ax=self.ax,
        )
        hep.histplot(
            bkg_histogram,
            bins=self.bin_edges,
            histtype="step",
            label="Background",
            ls=linestyle,
            color="blue",
            ax=self.ax,
        )
        self.ax.legend(prop={"size": 28})

    def plot(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlabel(r"$\mathcal{D}_{\tau}$", fontdict={"size": 28})
        ax.set_yscale("log")
        ax.set_ylabel("Relative yield / bin")
        return fig, ax

    def save(self, output_path: str):
        self.fig.savefig(output_path, format="pdf")
        plt.close("all")


class TaggerMultiEvaluator:
    def __init__(self, output_dir: str, cfg: DictConfig):
        self.output_dir = output_dir
        self.cfg = cfg
        os.makedirs(self.output_dir, exist_ok=True)
        self.metrics = list(self.cfg.metrics.tagging.metrics.keys())

        self.tagging_plots = {}
        self.efficiency_plots = {
            metric: EfficiencyPlot(self.cfg, metric) for metric in self.metrics
        }
        self.fakerate_plots = {
            metric: FakeRatePlot(self.cfg, metric) for metric in self.metrics
        }
        self.roc_plot = ROCPlot(self.cfg)
        self.wp_values = {}

    def combine_results(self, evaluators: list):
        for evaluator in evaluators:
            self.tagging_plots[evaluator.algorithm] = TauClassifierPlot()
            self.tagging_plots[evaluator.algorithm].add_line(evaluator, "test")
            self.roc_plot.add_line(evaluator)
            for metric in self.metrics:
                self.efficiency_plots[metric].add_line(evaluator)
                self.fakerate_plots[metric].add_line(evaluator)
        self._get_wp_values(evaluators)

    def _get_wp_values(self, evaluators):
        for evaluator in evaluators:
            self.wp_values[evaluator.algorithm] = {
                "Loose": evaluator.loose_wp,
                "Medium": evaluator.medium_wp,
                "Tight": evaluator.tight_wp,
            }

    def save(self):
        for metric in self.metrics:
            fr_output_path = os.path.join(self.output_dir, f"{metric}_fakerates.pdf")
            self.fakerate_plots[metric].save(fr_output_path)
            eff_output_path = os.path.join(
                self.output_dir, f"{metric}_efficiencies.pdf"
            )
            self.efficiency_plots[metric].save(eff_output_path)
        roc_output_path = os.path.join(self.output_dir, f"ROC.pdf")
        self.roc_plot.save(roc_output_path)
        for algorithm in self.tagging_plots.keys():
            cls_output_path = os.path.join(
                self.output_dir, f"classifier_scores_{algorithm}.pdf"
            )
            self.tagging_plots[algorithm].save(cls_output_path)

        wp_values_path = os.path.join(self.output_dir, "WP_values.json")
        with open(wp_values_path, "wt") as out_file:
            json.dump(self.wp_values, out_file, indent=4, cls=NpEncoder)


# Example usage:
#     te1 = TaggerEvaluator(
#         signal_predictions=sig_data.binary_classification.pred,
#         signal_truth=sig_data.binary_classification.target,
#         signal_gen_tau_p4=sig_info_data.gen_jet_tau_p4s,
#         signal_reco_jet_p4=sig_info_data.reco_jet_p4s,
#         bkg_predictions=bkg_data.binary_classification.pred,
#         bkg_truth=bkg_data.binary_classification.target,
#         bkg_gen_jet_p4=bkg_info_data.gen_jet_p4s,
#         bkg_reco_jet_p4=bkg_info_data.reco_jet_p4s,
#         cfg=cfg.tagging,
#         sample="ZH",
#         algorithm="DeepSet"
#     )
#     te2 = TaggerEvaluator(
#         signal_predictions=sig_data.binary_classification.pred,
#         signal_truth=sig_data.binary_classification.target,
#         signal_gen_tau_p4=sig_info_data.gen_jet_tau_p4s,
#         signal_reco_jet_p4=sig_info_data.reco_jet_p4s,
#         bkg_predictions=bkg_data.binary_classification.pred,
#         bkg_truth=bkg_data.binary_classification.target,
#         bkg_gen_jet_p4=bkg_info_data.gen_jet_p4s,
#         bkg_reco_jet_p4=bkg_info_data.reco_jet_p4s,
#         cfg=cfg.tagging,
#         sample="ZH",
#         algorithm="DeepSet"
#     )
#     tme = TaggerMultiEvaluator("output_plots", cfg)
#     tme.combine_results([te1, te2])
#     tme.save_results()
