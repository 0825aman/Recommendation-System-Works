# KuaiRec 2.0 — Data Dictionary

This document describes every column in each Phase 2 file, its data type in the
cleaned Parquet output, and its meaning.

---

## small_matrix

The main user–video interaction log. Each row represents one viewing event.

| Column | Parquet type | Description |
|---|---|---|
| `user_id` | `Int64` | Unique user identifier. Range: 0–7175. |
| `video_id` | `Int64` | Unique video identifier. |
| `play_duration` | `Int64` | Duration the video was played, in milliseconds. |
| `video_duration` | `Int64` | Full duration of the video, in milliseconds. |
| `time` | `Datetime` | Timestamp of the interaction. **3.89% null** — source data gap. Parsed from `%Y-%m-%d %H:%M:%S%.f`. |
| `date` | `Int64` | Date of interaction as an integer (YYYYMMDD format). **3.89% null** — source data gap. |
| `timestamp` | `Float64` | Unix timestamp (seconds since epoch). **3.89% null** — source data gap. |
| `watch_ratio` | `Float64` | `play_duration / video_duration`. Values > 1.0 indicate replays and are **valid**. Never null. |

### watch_ratio Notes

- Values between 0.0 and 1.0: partial watch
- Values exactly 1.0: full watch
- Values > 1.0: replay (user watched beyond the video's end, looped back)
- Maximum observed: 571.52 — extreme outlier, preserved, flagged for Phase 3
- Minimum valid: 0.0 — no negative values present

---

## user_features

One row per unique user (7,176 users). Provides demographic and engagement signals.

| Column | Parquet type | Description |
|---|---|---|
| `user_id` | `Int64` | Unique user identifier. Primary key. |
| `user_active_degree` | `String` | Activity level: `full_active`, `high_active`, `middle_active`, `UNKNOWN`. |
| `is_lowactive_period` | `Boolean` | True if the user is in a low-activity period. |
| `is_live_streamer` | `Boolean` | True if the user produces live streams. |
| `is_video_author` | `Boolean` | True if the user produces regular video content. |
| `follow_user_num` | `Int64` | Number of users this user follows. |
| `follow_user_num_range` | `String` | Binned range of `follow_user_num` (e.g., `"(0,10]"`). |
| `fans_user_num` | `Int64` | Number of followers this user has. |
| `fans_user_num_range` | `String` | Binned range of `fans_user_num`. |
| `friend_user_num` | `Int64` | Number of mutual follows (friends). |
| `friend_user_num_range` | `String` | Binned range of `friend_user_num`. |
| `register_days` | `Int64` | Days since account registration. |
| `register_days_range` | `String` | Binned range of `register_days`. |
| `onehot_feat0` | `Boolean` | Anonymized binary feature 0. |
| `onehot_feat1` | `Int64` | Anonymized categorical feature 1. |
| `onehot_feat2` | `Int64` | Anonymized categorical feature 2. |
| `onehot_feat3` | `Int64` | Anonymized categorical feature 3. |
| `onehot_feat4` | `Int64` | Anonymized categorical feature 4. |
| `onehot_feat5` | `Int64` | Anonymized categorical feature 5. |
| `onehot_feat6` | `Int64` | Anonymized categorical feature 6. |
| `onehot_feat7` | `Int64` | Anonymized categorical feature 7. |
| `onehot_feat8` | `Int64` | Anonymized categorical feature 8. |
| `onehot_feat9` | `Int64` | Anonymized categorical feature 9. |
| `onehot_feat10` | `Int64` | Anonymized categorical feature 10. |
| `onehot_feat11` | `Int64` | Anonymized categorical feature 11. |
| `onehot_feat12` | `Boolean` | Anonymized binary feature 12. |
| `onehot_feat13` | `Boolean` | Anonymized binary feature 13. |
| `onehot_feat14` | `Boolean` | Anonymized binary feature 14. |
| `onehot_feat15` | `Boolean` | Anonymized binary feature 15. |
| `onehot_feat16` | `Boolean` | Anonymized binary feature 16. |
| `onehot_feat17` | `Boolean` | Anonymized binary feature 17. |

---

## user_features_raw

Companion to `user_features`. Contains original (pre-binned) values and additional
device/app-install signals. Same 7,176 users as `user_features`.

Notable columns (beyond those shared with `user_features`):

| Column | Parquet type | Description |
|---|---|---|
| `user_id` | `Int64` | Primary key. |
| `is_lowactive_period` | `Boolean` | Same definition as in `user_features`. |
| `is_live_author` | `Boolean` | True if the user has authored live content. |
| `is_photo_author` | `Boolean` | True if the user has authored photo posts. |
| `is_install_douyin` | `Boolean` | Has the Douyin (TikTok China) app installed. |
| `is_install_huoshan` | `Boolean` | Has the Huoshan app installed. |
| `is_install_xigua` | `Boolean` | Has the Xigua video app installed. |
| `is_install_douyu` | `Boolean` | Has the Douyu streaming app installed. |
| `is_install_huya` | `Boolean` | Has the Huya streaming app installed. |
| `is_install_yy` | `Boolean` | Has the YY live app installed. |
| `os_version` | `String` | Device OS version string. Cast from mixed int/string source. |

---

## item_categories

One row per video (3,897 videos). Contains video category tags.

| Column | Parquet type | Description |
|---|---|---|
| `video_id` | `Int64` | Unique video identifier. Primary key. |
| `feat` | `String` | Original category list as a Python-style string: `"[27, 9]"`. Preserved for provenance. |
| `feat_parsed` | `List(Int64)` | Parsed integer list from `feat`. Use this column for analysis. |

### Category Tags

Category integers (e.g., `27`, `9`) are opaque identifiers from Kuaishou's internal taxonomy.
Their semantic meaning (e.g., "pet", "food") is documented in `kuairec_caption_category`
via `first_level_category_id` / `first_level_category_name`.

---

## kuairec_caption_category

One row per video (3,897 videos). Contains captions and a three-level category hierarchy.

| Column | Parquet type | Description |
|---|---|---|
| `video_id` | `Int64` | Unique video identifier. Primary key. |
| `manual_cover_text` | `String` | Manually annotated cover text. May be in Chinese. |
| `caption` | `String` | Auto-generated video caption. **12.63% null** — expected, not all videos have captions. |
| `topic_tag` | `String` | Topic tags as a string list. |
| `first_level_category_id` | `Int64` | Top-level category identifier. |
| `first_level_category_name` | `String` | Top-level category name (e.g., "pet", "food", "sport"). |
| `second_level_category_id` | `Int64` | Mid-level category identifier. |
| `second_level_category_name` | `String` | Mid-level category name (e.g., "cat", "snack"). |
| `third_level_category_id` | `Int64` | Fine-grained category identifier. |
| `third_level_category_name` | `String` | Fine-grained category name (e.g., "cute_cat", "chips"). |

### Null Captions

12.63% of videos have null `caption` values. This is expected — the dataset's auto-caption
pipeline did not process all videos. These nulls are preserved. Text imputation or
exclusion strategies will be decided in Phase 3.

---

## Referential Integrity

| Relationship | Status |
|---|---|
| `small_matrix.user_id` → `user_features.user_id` | ✅ All user IDs in matrix found in features |
| `small_matrix.video_id` → `item_categories.video_id` | ✅ All video IDs in matrix found in categories |
| `kuairec_caption_category.video_id` → `item_categories.video_id` | ✅ All caption video IDs found in categories |
| `user_features.user_id` == `user_features_raw.user_id` | ✅ Both have exactly 7,176 users |
