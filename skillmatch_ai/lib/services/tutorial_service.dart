// lib/services/tutorial_service.dart

import 'package:shared_preferences/shared_preferences.dart';

class TutorialService {
  static const _key = 'skillmatch_tutorial_done';

  /// Returns true if this is the user's first launch (tutorial not yet seen).
  static Future<bool> isFirstLaunch() async {
    final prefs = await SharedPreferences.getInstance();
    return !(prefs.getBool(_key) ?? false);
  }

  /// Mark the tutorial as completed.
  static Future<void> markDone() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_key, true);
  }

  /// Reset so the tutorial shows again (used by the info button).
  static Future<void> reset() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_key);
  }
}
