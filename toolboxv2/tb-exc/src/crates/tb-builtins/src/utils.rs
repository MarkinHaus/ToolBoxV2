//! Utility functions for TB Language
//!
//! - JSON/YAML parsing and serialization
//! - Time and timezone handling
//! - String manipulation

use crate::error::{BuiltinError, BuiltinResult};
use serde_json::Value as JsonValue;
use serde_yaml::Value as YamlValue;
use chrono::{DateTime, Local, Utc, TimeZone, Datelike, Timelike, Offset};
use chrono_tz::Tz;
use std::collections::HashMap;

// ============================================================================
// JSON UTILITIES
// ============================================================================

/// Parse JSON string to dictionary
pub fn json_parse(json_str: &str) -> BuiltinResult<JsonValue> {
    serde_json::from_str(json_str)
        .map_err(|e| BuiltinError::Serialization(format!("JSON parse error: {}", e)))
}

/// Convert dictionary to JSON string
pub fn json_stringify(value: &JsonValue, pretty: bool) -> BuiltinResult<String> {
    if pretty {
        serde_json::to_string_pretty(value)
    } else {
        serde_json::to_string(value)
    }
    .map_err(|e| BuiltinError::Serialization(format!("JSON stringify error: {}", e)))
}

/// Parse JSON from bytes
pub fn json_parse_bytes(data: &[u8]) -> BuiltinResult<JsonValue> {
    serde_json::from_slice(data)
        .map_err(|e| BuiltinError::Serialization(format!("JSON parse error: {}", e)))
}

/// Convert value to JSON bytes
pub fn json_to_bytes(value: &JsonValue) -> BuiltinResult<Vec<u8>> {
    serde_json::to_vec(value)
        .map_err(|e| BuiltinError::Serialization(format!("JSON serialize error: {}", e)))
}

// ============================================================================
// YAML UTILITIES
// ============================================================================

/// Parse YAML string to dictionary
pub fn yaml_parse(yaml_str: &str) -> BuiltinResult<YamlValue> {
    serde_yaml::from_str(yaml_str)
        .map_err(|e| BuiltinError::Serialization(format!("YAML parse error: {}", e)))
}

/// Convert dictionary to YAML string
pub fn yaml_stringify(value: &YamlValue) -> BuiltinResult<String> {
    serde_yaml::to_string(value)
        .map_err(|e| BuiltinError::Serialization(format!("YAML stringify error: {}", e)))
}

/// Parse YAML from bytes
pub fn yaml_parse_bytes(data: &[u8]) -> BuiltinResult<YamlValue> {
    serde_yaml::from_slice(data)
        .map_err(|e| BuiltinError::Serialization(format!("YAML parse error: {}", e)))
}

// ============================================================================
// TIME UTILITIES
// ============================================================================

/// Time information dictionary
#[derive(Debug, Clone)]
pub struct TimeInfo {
    pub year: i32,
    pub month: u32,
    pub day: u32,
    pub hour: u32,
    pub minute: u32,
    pub second: u32,
    pub microsecond: u32,
    pub weekday: u32,
    pub timezone: String,
    pub offset: i32,
    pub timestamp: i64,
    pub iso8601: String,
}

impl TimeInfo {
    pub fn to_hashmap(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("year".to_string(), self.year.to_string());
        map.insert("month".to_string(), self.month.to_string());
        map.insert("day".to_string(), self.day.to_string());
        map.insert("hour".to_string(), self.hour.to_string());
        map.insert("minute".to_string(), self.minute.to_string());
        map.insert("second".to_string(), self.second.to_string());
        map.insert("microsecond".to_string(), self.microsecond.to_string());
        map.insert("weekday".to_string(), self.weekday.to_string());
        map.insert("timezone".to_string(), self.timezone.clone());
        map.insert("offset".to_string(), self.offset.to_string());
        map.insert("timestamp".to_string(), self.timestamp.to_string());
        map.insert("iso8601".to_string(), self.iso8601.clone());
        map
    }
}

/// Get current time information
///
/// # Arguments
/// * `timezone` - Optional timezone (e.g., "America/New_York", "Europe/London")
///                If None, uses local timezone
pub fn get_time(timezone: Option<String>) -> BuiltinResult<TimeInfo> {
    let now = Utc::now();

    let (dt, tz_name, offset) = if let Some(tz_str) = timezone {
        if tz_str.to_lowercase() == "auto" || tz_str.to_lowercase() == "local" {
            let local = Local::now();
            let offset = local.offset().local_minus_utc();
            (local.naive_local(), "Local".to_string(), offset)
        } else {
            let tz: Tz = tz_str.parse()
                .map_err(|_| BuiltinError::InvalidArgument(
                    format!("Invalid timezone: {}", tz_str)
                ))?;
            let dt_tz = now.with_timezone(&tz);
            let offset = dt_tz.offset().fix().local_minus_utc();
            (dt_tz.naive_local(), tz_str, offset)
        }
    } else {
        let local = Local::now();
        let offset = local.offset().local_minus_utc();
        (local.naive_local(), "Local".to_string(), offset)
    };

    let dt_utc = Utc.from_utc_datetime(&dt);
    Ok(TimeInfo {
        year: dt.year(),
        month: dt.month(),
        day: dt.day(),
        hour: dt.hour(),
        minute: dt.minute(),
        second: dt.second(),
        microsecond: dt_utc.timestamp_subsec_micros(),
        weekday: dt.weekday().num_days_from_monday(),
        timezone: tz_name,
        offset,
        timestamp: dt_utc.timestamp(),
        iso8601: dt_utc.to_rfc3339(),
    })
}

/// Format time as string
pub fn format_time(time_info: &TimeInfo, format: &str) -> String {
    // Simple format implementation
    format
        .replace("%Y", &time_info.year.to_string())
        .replace("%m", &format!("{:02}", time_info.month))
        .replace("%d", &format!("{:02}", time_info.day))
        .replace("%H", &format!("{:02}", time_info.hour))
        .replace("%M", &format!("{:02}", time_info.minute))
        .replace("%S", &format!("{:02}", time_info.second))
        .replace("%Z", &time_info.timezone)
}

/// Parse time from string
pub fn parse_time(time_str: &str, format: Option<&str>) -> BuiltinResult<TimeInfo> {
    let dt = if let Some(fmt) = format {
        DateTime::parse_from_str(time_str, fmt)
            .map_err(|e| BuiltinError::InvalidArgument(
                format!("Time parse error: {}", e)
            ))?
            .with_timezone(&Utc)
    } else {
        // Try common formats
        DateTime::parse_from_rfc3339(time_str)
            .or_else(|_| DateTime::parse_from_rfc2822(time_str))
            .map_err(|e| BuiltinError::InvalidArgument(
                format!("Time parse error: {}", e)
            ))?
            .with_timezone(&Utc)
    };

    let local = dt.with_timezone(&Local);
    let offset = local.offset().local_minus_utc();

    Ok(TimeInfo {
        year: local.year(),
        month: local.month(),
        day: local.day(),
        hour: local.hour(),
        minute: local.minute(),
        second: local.second(),
        microsecond: local.timestamp_subsec_micros(),
        weekday: local.weekday().num_days_from_monday(),
        timezone: "Local".to_string(),
        offset,
        timestamp: local.timestamp(),
        iso8601: local.to_rfc3339(),
    })
}

/// Get timestamp (seconds since epoch)
pub fn get_timestamp() -> i64 {
    Utc::now().timestamp()
}

/// Get timestamp in milliseconds
pub fn get_timestamp_ms() -> i64 {
    Utc::now().timestamp_millis()
}

/// Sleep for specified duration (async)
pub async fn sleep(seconds: f64) {
    let duration = std::time::Duration::from_secs_f64(seconds);
    tokio::time::sleep(duration).await;
}

// ============================================================================
// STRING UTILITIES
// ============================================================================

/// Convert bytes to hex string
pub fn bytes_to_hex(data: &[u8]) -> String {
    data.iter()
        .map(|b| format!("{:02x}", b))
        .collect()
}

/// Convert hex string to bytes
pub fn hex_to_bytes(hex: &str) -> BuiltinResult<Vec<u8>> {
    (0..hex.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&hex[i..i + 2], 16)
                .map_err(|e| BuiltinError::InvalidArgument(
                    format!("Invalid hex string: {}", e)
                ))
        })
        .collect()
}

/// Base64 encode
pub fn base64_encode(data: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(data)
}

/// Base64 decode
pub fn base64_decode(encoded: &str) -> BuiltinResult<Vec<u8>> {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .map_err(|e| BuiltinError::InvalidArgument(
            format!("Base64 decode error: {}", e)
        ))
}

// Add base64 dependency
use base64;

