import en from "./i18n/en.json";
import fa from "./i18n/fa.json";
import ar from "./i18n/ar.json";

export type Locale = "en" | "fa" | "ar";
type TranslationCatalog = Record<string, string>;

const catalogs: Record<Locale, TranslationCatalog> = { en, fa, ar };

export function createTranslator(
  locale: Locale
): (key: string, values?: Record<string, string | number>) => string {
  return (key, values = {}) => {
    const message = catalogs[locale][key] ?? catalogs.en[key] ?? key;
    return Object.entries(values).reduce(
      (result, [name, value]) => result.replaceAll(`{${name}}`, String(value)),
      message
    );
  };
}

export function normalizeLocale(value: unknown): Locale {
  return value === "fa" || value === "ar" ? value : "en";
}
