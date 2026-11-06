#!/usr/bin/env python3
"""Check that everything the pending HPC runs need is present and current. Read-only; run on the HPC login node.

Usage:  python3 verify_hpc_ready.py            (defaults to ~/DriveVLA-Safety-RL-offline-20260907)
"""
import hashlib, json, os, subprocess, sys
from pathlib import Path

OFF = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else Path("~/DriveVLA-Safety-RL-offline-20260907").expanduser()
W0 = Path("~/DriveVLA-W0").expanduser()
ASSET = Path("/gpfs/accounts/xinxia_root/xinxia0/huamin/DriveVLA-W0-data")
COHORT = ASSET / "pretrained_models/DriveVLA-W0/posttrain_runs/rl_pareto_navtest_v2_20261005_133703_37354"
RUN = OFF / "output/offline_scaled_20261008_150131"
EXPECTED = {
 "tools/rl_prep/offline_verified_data.py": "98005fa438c3fec6faa7576494a954b3df36f510ef71276e049f4bd12a797a56",
 "tools/rl_prep/train_offline_verified_policy.py": "5120d638fa27b255399921046247a481e536d1aefb6641cbf519ecaa2c9e37be",
 "tools/rl_prep/pair_preference.py": "dd508e4ff8a625544f924b44879190c6ba62dad7d1782a742ce99d830e342e4a",
 "tools/rl_prep/build_margin_preferences.py": "793d6e309f855ea17491cee46e2aa48edfac8dad2fbf6c581b4095f2f642bc40",
 "tools/rl_prep/build_log_disjoint_manifest.py": "644707113897410d535e40595cd44705b492fc13be88a555b993535f660de056",
 "tools/rl_prep/analyze_trajectory_selector.py": "5b0a31e14052223f1c5fef5f1c7163c7f0ea6a90892ab8b3716bd5c7be76b7bf",
 "tools/rl_prep/fit_and_apply_selector.py": "a281d3d973213ad85777d1ef455b7b044a9289db3dfb160220edbc9df2035644",
 "tools/rl_prep/check_navtest_collection.py": "dd091ea741b062d77a2362293c3062efc481fe57c39472a98133c201204e3ae7",
 "tools/rl_prep/selector_on_eval_tokens.py": "3bd7bdefa4d98ca1a91ae18f56305c691a215f35941a421718b5ae2af8411c65",
 "tools/rl_prep/summarize_offline_scaled.py": "09ce4f05e24520da34a50e633d2dafe494bb541a9b3bbea264a8ab8b698e14e3",
 "tools/rl_prep/show_prefs_reports.py": "4c37ee0e716a331e48c4add1d372de103654770e7e53f610433f65e8935ac854",
 "tools/rl_prep/geometric_features.py": "3aef4a5e8740c504d7b3f8d602abb08867fce6adafe4750c304397e15a8b992b",
 "tools/rl_prep/extract_geometric_features.py": "d77d2788f7f64778d4369a4e9d6b2b1c0fa39b59ec8f3db86d3020bdfb9b2b73",
 "tools/rl_prep/test_pair_preference.py": "746facb1364dff249ef36f10f634f143001b3527678fa54aa7cc5970069fcc8b",
 "tools/rl_prep/test_build_margin_preferences.py": "65c51814f8f2c0d2813bc5d75ae1041d9d564f275159164412f456bb772a8853",
 "tools/rl_prep/test_fit_and_apply_selector.py": "2a67c3a26a413d759e96f84825cca42a28032b6adc6dd3afe90408ccab435cee",
 "tools/rl_prep/test_selector_on_eval_tokens.py": "b7a0988f00159305eb7d8f935feeec1621705566f688b614f8e82b7b8ffe1b22",
 "tools/rl_prep/test_analyze_trajectory_selector.py": "76fb87fa9020e44381d15652440f5682108ef1fd94bbb9e06f0886235ba1ccca",
 "tools/rl_prep/test_log_disjoint_offline.py": "63bce03f4fbf130fe459bbe663382bf591cf941ac49103a4ec37110fc2dd6d99",
 "tools/rl_prep/test_summarize_offline_scaled.py": "b96f25360de5056246cb8512b37b194ea5aee1ca6307fd46bd7e565551bcac7b",
 "scripts/hpc/submit_navtest_candidates.slurm": "393d74b5f4442b782f6833247e9591c824f06b25858da9cd9160ad4b791c7842",
 "scripts/hpc/run_navtest_selector_e2e.sh": "9b2cc68ed4963774d1ea2c4a627baa2956d8bf3ec3f231af754f5f826424dee4",
 "scripts/hpc/submit_pair_pref.sh": "11fcc51bbe38a6dc8dc6ed8c3b44b20a206416ebaa60d24e0a9018e2f0a6a0cf",
 "scripts/hpc/submit_pair_pref_train.slurm": "6f1c7c9618a7b2b1b856ad887d332ae5c4acb31b8e7ca079890237f69f3c2770",
 "scripts/hpc/submit_offline_scaled.sh": "4f986f347b8cbb99279cb2861a70256f3e107d98585b17fcd906f200f2bcf5d2",
 "scripts/hpc/submit_offline_scaled_train.slurm": "5fa02db5dc8f225210b2bf17bd25fe3e0dc80a1a0531d4b6cdd59855d37bfa51",
 "scripts/hpc/submit_extract_geometric_features.slurm": "7e902c16fc69ee45377d8978b7620b20317bdcb1be307f987c7739eaadb4d872",
 "tools/rl_prep/extract_wm_pca_features.py": "472d18f6c7c0f396c9b623c7f47dd48dbcdfc88716db98c40ef9c732eb832424",
 "tools/rl_prep/test_extract_wm_pca_features.py": "956b269c88ca127f270ebf0895a31822c250e67147b422e1bc3e5a010750f22f",
 "tools/rl_prep/test_geometric_features.py": "be2133dfdb04b5cf33a1090025443f486dd5a03ec98f809e531dfe155249ac1f",
 "scripts/hpc/submit_wm_pca_features.slurm": "dad81d27da072391d0cc291ef2798e96d5e6df98cfe3c6da963a7e875f1865d4",
 "scripts/hpc/submit_selector_extras.slurm": "e2a6386ecc41f186a652cda830b42cff5781d312a08ff77af9847363d25d83f5",
 "scripts/hpc/submit_selector_upgrade.sh": "d4249a048da7f385531ac173f4533357b95e79e480af63c0d7afe80c84da99c9",
 "tools/rl_prep/test_check_navtest_collection.py": "01339e62d0c7c599ad92e4af3b900763f2fd049ad422d2d3ef8b551df6798741",
 "tools/rl_prep/analyze_probe_vs_eval.py": "667926d2497bde60fdee7f5447c4695ef71fdada2a3fb0a4bb02ebb284492bc0",
 "tools/rl_prep/test_analyze_probe_vs_eval.py": "b11ec952691c81b25325023433d30a955bfc8249caf6f3ce2ebaef797fbcfc0b",
 "scripts/hpc/run_navtest_rerun.sh": "7a82ac06c47c003bf3855b06ff91f3dcbcf1c8bd12f4a6957acb1b1893cfdd17",
 "tools/rl_prep/eval_action_policy.py": "da07b17b78158565098d32e262d43bc84f1881bf0765551d39a309e90ccf2c99",
 "tools/rl_prep/summarize_policy_eval.py": "977d9bb964e4ce818811c1dacf8eafea96f73d6ff8fb15ff5b7b40b85b504336",
 "tools/rl_prep/test_summarize_policy_eval.py": "3599a430deaf5b229691323c8661d45c87539ef45f5e7bec47cabbe12d7ee469",
 "scripts/hpc/submit_navtest_policy_eval.slurm": "ef7b28ff000b3c7c576934e5a62bfb69bbd38a5df2c78ff238bb6a678cdd9b81",
 "scripts/hpc/submit_navtest_policy_eval.sh": "0ef7e7bf0dde835458e7d0ecb74cd7cabf9e1140d3edad98c427ddc75dc33544",
 "scripts/hpc/submit_pair_pref_more_seeds.sh": "6cf0f354899bfe99e52c4fa7d1b2aebdead69f70f7996ea8b3a1fe53f6f4b17e",
 "scripts/hpc/submit_selector_struct.slurm": "49ba2096937a70efb5dea30b4c641ba2ce4c50aee2e5f1291bae004ddac9a8bc",
 "tools/rl_prep/guidance_geometry.py": "c8fd4ce91a5c6691b3e8aea5985f05c7897379ab1e00f4bca0d3025f82cac51b",
 "tools/rl_prep/test_guidance_geometry.py": "5565e01ec3b03b483fe11377ce2e8da1710cba5791094ec4cd69df5fdf103c9c",
 "tools/rl_prep/extract_guidance_inputs.py": "69e5aaa9b2c1ab1cf80a2cba609a25fb94ef7b1926be7cc57f47c347dd453ac1",
 "tools/rl_prep/test_extract_guidance_inputs.py": "1a1a524f07445552184f4179cdcf81a07a5ea411076af22e525e300e001e2b10",
 "scripts/hpc/submit_extract_guidance_inputs.slurm": "8ef6d55f2a9b12826c15d4a25ae0a3c4493719e3e4121944aff7e91caea22bf9",
 "tools/rl_prep/collect_guided_candidates.py": "90739fa9e5d0305393b45f71d49d78babd075aaa20c400835660a7d21db79442",
 "tools/rl_prep/make_token_log_map.py": "679fb9a1c3a45a6e66bca77dee63a3a6990c0029facb36215c57e069325dbb5f",
 "tools/rl_prep/compare_guidance_runs.py": "bfbc9aeb9039ff0d707c21fd22a8e7f4b8fd218e9434420b1529e590611a4a87",
 "scripts/hpc/submit_guided_candidates.slurm": "dbfd155c2fbcc3bc04db9de7bb2228ee66fd94b9ce0b3459f8597b6bbef1856e",
 "scripts/hpc/run_guidance_dev.sh": "0d7580d5ec57645b9026e2b179d54180228741b7b84d8a3cf4727af1bd22b510",
 "tools/rl_prep/check_guidance_alignment.py": "8beb2e4b7f5cbe36845df54b420531d9c3f3e67f3d6fcfdcd92de30464d6bf12",
 "tools/rl_prep/perception_noise.py": "d8f3a8de328caa57f79f205d54dd6f043a1eef88fc07d6eeb10b522902709633",
 "tools/rl_prep/test_perception_noise.py": "42d6185d4e0652aa4a0049033fde3bda0f23f282718b0e78269077b4c00dcaef",
 "tools/rl_prep/summarize_noise_robustness.py": "b0188ab04b830afd55e71421da1968d6d1de8b0ae8bb4f5d819701f0d381e69a",
 "scripts/hpc/run_perception_noise.sh": "c136fd6a93d72d81783b3f982900adc883be46f8539aacafd73de3c6f0636d20",
 "tools/rl_prep/collect_anchor_candidates.py": "0af90ec215c7262c1675d7ff5cecb530b4cac09703b75e8b7ef69183944574cf",
 "scripts/hpc/submit_anchor_candidates.slurm": "9be002c6d4a9335705fb370e037de352ec9460f00d793db46dc8e8b45ffaac8f",
 "tools/rl_prep/analyze_guidance_headroom.py": "18ae6299dc1fdc8c6fc4951ccf9dc3c3bc5a59eaa4ea4687263f5ea2187ac4f4",
 "tools/rl_prep/test_analyze_guidance_headroom.py": "c71fa4143a0ddb2d286667b8e195bb0ba01820e3df1739e5d15d491b7f8e2127",
 "tools/rl_prep/select_hard_scenes.py": "2a84f1afbb0c184821ad8dc01beed7194bba91abf267bad649be108d865b3af1",
 "tools/rl_prep/analyze_guided_repair.py": "ef630a9fdfb0484ecd83e95cd71329d2ad49cf77bb8a5f62b6a1889ac6d214b7",
 "tools/rl_prep/test_analyze_guided_repair.py": "2873b5e468707245bf8a2915a213c01e3b7e394c8c8b015863c4985724c4619d",
 "scripts/hpc/run_guidance_navtest.sh": "6b242ca8b88129688eb34e52bb4e417ee4602f76227151bd73afaef9c14d8225",
 "tools/rl_prep/compare_selector_runs.py": "28d57585dd478328c9ca6a21f8a0667928a9b1c4655932503546758d213ce073",
 "tools/rl_prep/test_compare_selector_runs.py": "584c674c6cdaf4ca1d9319f7e9af2daeafd46ada66a1878d1392b43de422e0a5",
 "tools/rl_prep/analyze_guided_harm.py": "637e6e0ad60a037d6cb3f201ada3630ccc2f0dd5d03316e74de0fd26e69a2258",
 "scripts/hpc/run_anchor_full.sh": "e70c6eaade9c6fef636d41604cd31676a4756f084f035e5ab1b79573c7f98b61",
 "tools/rl_prep/analyze_fallback_guidance.py": "322635e04a5fb8f848fe5f6449b823bfc25c48a8049aa42510028b83f1bd0048",
 "tools/rl_prep/test_analyze_fallback_guidance.py": "702add4a8343fdb32cbf0aa9d0fa16669a2fd15e17c91c6b4445f8e0d05e6f24",
 "scripts/hpc/setup_navsim_v2.sh": "3a07041994e0265d1af3d238fea0f19896c62a31b4ade236c884f941e43599fc",
 "scripts/hpc/setup_navsim_v2.slurm": "74e85d53735a29127367867da7990756694b30067307b32eba5ae08d4c2b09d6",
 "tools/rl_prep/build_ledger.py": "0a353bc0002b01f1713c521cf5c26a22c889c90106e317994f378de369d7ac55",
 "tools/rl_prep/test_build_ledger.py": "7b01ed179cd4e1969c2fa66943df6bb4589c3a8465a50e8a55e95c62eb8d5be7",
 "tools/rl_prep/compare_within_selector.py": "20bd51b03a6713d427965d1a046aec04d9bff608520a67281d7b28c176f4da83",
 "tools/rl_prep/test_compare_within_selector.py": "7fde5ffec7cf18ea478c69e58090b9ae3f4f98fa169cd31550efd6faad76b8dd",
 "scripts/hpc/setup_navsim_v2_clone.slurm": "b598e01d88c5ee021cf10561692e18dbe97ac8b435e4e8a44f8a9c9dfe50fbf9",
 "tools/rl_prep/budgeted_awfm.py": "4a63bef16629453317c9208a29ea181b446684190c42ed1c75d7afda72f16edd",
 "tools/rl_prep/analyze_repair_transfer.py": "d73b58cf70e5157ee89e4d71473dffdcdcae68a829bff9405227dd310f227b43",
 "tools/rl_prep/test_budgeted_awfm.py": "3daefe64f81ff8c7db158c8f283c244aa7a1353c2cffa942c165e22187bbf948",
 "scripts/hpc/submit_awfm.sh": "00681f2107eb31fd1944fff9690b7c77d7fe1f0aa39d23ec023206d24a4c3f12",
 "scripts/hpc/submit_awfm_train.slurm": "995f2143b91b8723c80afdc3f7da792c907282d155c0379b6aea95e04c9e72a2",
 "tools/rl_prep/test_analyze_repair_transfer.py": "593405ce20ec50b35f477afca24c3599ee2e7a998586d9e15a77957054446a34",
 "tools/rl_prep/awfm_statistics.py": "7a416313c4b706321eb5dbfc725ecd585be9e2683ff6c6ccc09ec4b95e01ba9e",
 "tools/rl_prep/test_offline_verified_training.py": "35611f29fe8a996bc4babd682e3e5870425667dd7e522ca86c2974a80dbfad4a",
 "tools/rl_prep/awfm_preservation.py": "ab9919a355ec22bad21acc6fad1eda9c5bb424a48e985b26aceb530ff036a246",
 "tools/rl_prep/check_awfm_run.py": "b45cb6adefd9b773125c7c787f5bebe8542e1a7c3f2730a55949492587c6a8fa",
 "tools/rl_prep/test_awfm_validation.py": "c644618211b5ac6a0fb9ce62b7ff23556406f5eb7e4799bef7c6d5a1e4ae5fa1",
 "tools/rl_prep/check_awfm_inputs.py": "758316d98a9b4d42b1426d2433c3c7ff70114e766f66263820f30d275ec70eac",
 "tools/rl_prep/test_awfm_preservation.py": "ed998a4b67e9dea4e165a00a1bcb93f8754a9a5adc430bcebc856a8a5e2c00ca",
 "docs/awfm_v2_runbook.md": "9bef4a9d5b90129af475854c0ffd4b01a6dcdb593bc814df396088866e872ff9",
 "scripts/hpc/submit_awfm_summary.slurm": "5937cd0199dbe4a26907b293143c99e799bf2e8a603fa71648d496194a1da5ec",
 "scripts/hpc/submit_official_navtest_cache.slurm": "8d2090f87171b722c5bdaed617e7eadf785686e82b3c23634997924067b94316",
 "scripts/hpc/submit_official_navtest_infer.slurm": "f4e396694457007cc0cfb01b90faa491b10243160f9d2435206e9e775718363f",
 "scripts/hpc/submit_official_navtest_score.slurm": "cefaeacf0ab286d077aae91dbc3198be9a12897e3d260ba86e740ee818db3e74",
 "scripts/hpc/fetch_released_models.py": "958c14eeea35c487521bb65a56d0c9cfdf19a0114d9975f7a7ab3d9c88db6cbe",
 "scripts/hpc/submit_fetch_released_models.slurm": "fe6369604fcfe1b0a648eb2a4862d34ad8db5a520bc1db919591ee0952a856d0",
 "scripts/hpc/run_inference_anchor_patch.py": "337166def68d21d6b0cb6db192da74ce60589e9fc806126644ad4754e21dc6f7",
 "tools/rl_prep/prepare_official_cohort.py": "12dfca573576d2aa98bbde221b4e678251b22ea1a8b866615002fb3e5a94f317",
 "tools/rl_prep/test_prepare_official_cohort.py": "25acac2a5eeea0efda5ead3272eff85029d908b8a8334b97f4e9a3aed20256f1",
 "scripts/hpc/refresh_timestamps.slurm": "ede7c6e68a0218dd80b96c799ec6e3daa961231d2598fdda5e6c5944faa55f6d",
 "tools/rl_prep/export_selected_to_official_json.py": "b61f8a8fb0d699afbcdb22e089cee90d27c4c9428d0bba4b42dc2735f359277e",
 "tools/rl_prep/test_export_selected_to_official_json.py": "9871934e66e7b552edd5fdb4b3afb9a51557a361ad67e8486004c9cc1da33f09",
 "scripts/hpc/run_ttc_block_navtest.sh": "3f334bf78952c76f469c605f2613f7391e332a25e92dbc0b70373be2e490fc90",
 "scripts/hpc/run_anchor_navtest_full.sh": "0f593092d979aded58ec7bfa71240d37fa208a28d6bd1e403e8b16f8b36c8884",
 "scripts/hpc/run_map_noise_navtest.sh": "82a7605d60783a82a857bc8e2a7a5026985b2dd406f341dcac63edb9d1a14188",
 "tools/rl_prep/merge_policy_eval_shards.py": "12bea6cffef2b13fa394ef12532fa3ab61d091e5135bdc0c4d2c7838f21bd719",
 "tools/rl_prep/test_merge_policy_eval_shards.py": "541c8c3f48676355d73b3784a114220966cf92bfb62140c6fab271351ea82b02",
 "scripts/hpc/submit_awfm_highbudget.sh": "49089eb6626f20ad5dd30c10a1af165c71466aad4459872af5a170fc8e082bb5",
 "tools/rl_prep/extract_policy_vlm_context.py": "62e9c63f489bebe219e8721c99d719fd17fe178522868b7f8421b03c28e226ad",
 "tools/rl_prep/make_policy_visual_blocks.py": "0c0f0baba2bbf617f4ed287e77d74f51a9df449944d7558be81c80d63a371479",
 "tools/rl_prep/test_make_policy_visual_blocks.py": "5590a4dc027514db619407aec5b0c71850fa6899f11bfc47c7eba31a3574af2b",
 "scripts/hpc/submit_policy_vlm_context.slurm": "db5d75cbce52b3ba490ca267103190d0361ee62c004a351037f17a41dbd3480d",
 "scripts/hpc/run_policy_visual_test.sh": "9a6f7e74c1b842af7732c44a1264759b41b11148f71b1074f65368fca6624983",
 "tools/rl_prep/probe_policy_context.py": "195af02698cee58a39338a00cb81d8d5852d0ee2e75db7f16d9f035ae6d52a72",
 "tools/rl_prep/test_probe_policy_context.py": "ed05b2c0cec95c66739f1923cafa20a47532e342a16c400c9185524a38be8c81",
 "tools/rl_prep/paired_scores_by_log.py": "16130239dea15b4392f893a3a8c346f826b33ccc05c8d4db2c87717316522222",
 "tools/rl_prep/test_paired_scores_by_log.py": "efbbcb050c5aa18da26d0ad9e258818c189d0b4f0b70560034e713565a7c6af2",
 "tools/rl_prep/vlm_token_layout.py": "39e01dd8c301008540613e93cfbf102888fb279f15466cb57e247bef843532d4",
 "tools/rl_prep/test_vlm_token_layout.py": "c7ee7aec5457d2c56ab8a2e95e4dc74c8cca64db089d602fdc4028f8290a53da",
 "tools/rl_prep/extract_policy_vlm_tokens.py": "e9a62c8422e9e7ecdc9ea1419db66b00282c1253302107b6143da1e1cc2d60e5",
 "scripts/hpc/submit_policy_vlm_tokens.slurm": "51e08143660d670060c54a2c573d93fd7e999d85aa3682cb907718e7c8464e0a",
 "tools/rl_prep/stageA_data.py": "f06cca15991a98ea100250cfb17316eea927e172174854fbc526e311e6e6ed72",
 "tools/rl_prep/stageA_model.py": "7f07ba9f939a962d5b61352580725d7ec3fba15d806919bbe8c368bbcd378ef7",
 "tools/rl_prep/test_stageA.py": "b3e40ca0e325faa5d02d8fed5f2ed4b116faac3add641de2e185361e53c4222d",
 "tools/rl_prep/test_stageA_model.py": "ee46b34c6830975d8fcf6d0ea642a348b3f6afdb63e60ba1e9eb7a2df601f995",
 "tools/rl_prep/stageA_run.py": "21f77934b2359a0449885e0a6abafd7ba6cf9aa8c27ff298ccf8a8791a99cf7b",
 "tools/rl_prep/stageA_compare.py": "59788110ba2360fdcdaa4b45cd4df83bba2a7f83552c3abeef5c6e6b8547fc60",
 "tools/rl_prep/stageA_pick_config.py": "bbb14eecfad7e7351ace0eb8ee1d90d7bb9366bffca9d350d7083553def5781f",
 "tools/rl_prep/test_stageA_run.py": "3548d30fba2c5b17fd7f126c86c29658aac555a2120af5df24a82b18305f2060",
 "tools/rl_prep/test_stageA_pick_config.py": "167b5178dd78ffdf07f828c046a2fb6ac0138c899764c2a1313ebed1310d1beb",
 "scripts/hpc/submit_stageA.slurm": "15a125855080405e9c3f391cfdee305cd1a8ba1b914b8fa0c5961496234b0b25"
}

problems = []
def line(status, text):
    print(f"{status:9s} {text}")
    if status not in ("OK", "INFO"):
        problems.append(text)

print("== 1. uploaded files (sha256 against the local versions)")
for rel, digest in EXPECTED.items():
    p = OFF / rel
    if not p.is_file():
        line("MISSING", rel)
    elif hashlib.sha256(p.read_bytes()).hexdigest() != digest:
        line("OUTDATED", rel + "   -> upload again")
    else:
        line("OK", rel)

print("\n== 2. prerequisites that are not uploaded")
def exists(path, what, kind="file"):
    ok = path.is_file() if kind == "file" else path.is_dir()
    line("OK" if ok else "MISSING", f"{what}: {path}")
    return ok
exists(RUN / "manifest_v2.json", "log-disjoint manifest")
for r in ("margin", "pdms", "nearest"):
    exists(OFF / f"output/prefs_v3_{r}/pairs.jsonl", f"preference pairs ({r})")
exists(OFF / "navsim_data/exp/world_action_full_v1", "candidate data (pool)", "dir")
for base in (OFF, W0):
    lat = base / "navsim_data/exp/world_action_latents_full_v1"
    nl = sum(1 for _ in lat.glob("*/*.npz")) if lat.is_dir() else 0
    line("INFO" if nl == 7171 else "INFO", f"world-model latents under {base.name}: {nl} files (expected 7171 in one of them; set WM_LATENT_ROOT accordingly)")
g = OFF / "navsim_data/exp/world_action_geom_full_v1"
n = sum(1 for _ in g.glob("*/*.npz")) if g.is_dir() else 0
line("OK" if n == 7171 else "MISSING", f"geometric features: {n} files (expected 7171)")
for f in ("test.pkl", "test_tokens.json", "token_log_map.json"):
    exists(COHORT / f, f"frozen navtest cohort {f}")
exists(COHORT / "cache", "frozen navtest metric cache", "dir")
exists(Path(os.path.expanduser("~/.conda/envs/navsim/bin/python")), "navsim python")
exists(OFF / "navsim_data/maps", "nuPlan maps", "dir")
exists(OFF / "navsim_data/exp/metric_cache_trainval_rl", "pool metric cache", "dir")

print("\n== 3. collection code used by the navtest end-to-end run (DriveVLA-W0)")
proto = W0 / "tools/rl_prep/rl_stage1_prototype.py"
if proto.is_file():
    hits = proto.read_text(errors="replace").count("COLLECT_ONLY")
    line("OK" if hits >= 1 else "MISSING", f"COLLECT_ONLY occurrences in {proto}: {hits}")
else:
    line("MISSING", str(proto))
for rel in ("pretrained_models/fast", "pretrained_models/Emu3-Stage1",
            "pretrained_models/DriveVLA-W0/Emu3_Flow_Matching_Action_Expert_PDMS_87.2"):
    exists(W0 / rel, "asset", "dir")
exists(W0 / "configs/normalizer_navsim_trainval/norm_stats.json", "normalizer stats")

print("\n== 4. python environment")
try:
    import sklearn, numpy
    line("OK", f"numpy {numpy.__version__}, scikit-learn {sklearn.__version__}")
except Exception as e:
    line("MISSING", f"python packages: {e}")
try:
    import torch
    line("OK", f"torch {torch.__version__}")
except Exception as e:
    line("INFO", f"torch not importable on this node ({e}); fine on GPU nodes if the env is the same")

print("\n== 5. output directories that must NOT already exist (the scripts refuse to overwrite)")
for rel in ("output/selector_on_600.json",):
    line("INFO" if (OFF / rel).exists() else "OK", f"{rel}: {'exists (already produced)' if (OFF / rel).exists() else 'not yet'}")
for name in ("smoke_pair_pref", "scaled_v1_marginpref_seed20", "scaled_v1_pdmspref_seed20", "scaled_v1_nearestpref_seed20"):
    line("OK" if not (RUN / name).exists() else "INFO", f"{name}: {'absent (good)' if not (RUN / name).exists() else 'ALREADY EXISTS - the launcher will refuse'}")

print("\n== summary")
if problems:
    print(f"{len(problems)} item(s) need attention:")
    for t in problems: print("  -", t)
    sys.exit(1)
print("everything the pending runs need is present and current")
