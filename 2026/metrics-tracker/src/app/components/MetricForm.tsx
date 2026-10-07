'use client';

import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Plus, X, Activity, Clock, Tag } from 'lucide-react';
import { addEntry, getSuggestions } from '@/app/actions';

export default function MetricForm() {
    const [metricName, setMetricName] = useState('');
    const [value, setValue] = useState('');
    const [dimensions, setDimensions] = useState<{ key: string, value: string }[]>([]);
    const [suggestions, setSuggestions] = useState<string[]>([]);
    const [showSuggestions, setShowSuggestions] = useState(false);
    const wrapperRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        function handleClickOutside(event: MouseEvent) {
            if (wrapperRef.current && !wrapperRef.current.contains(event.target as Node)) {
                setShowSuggestions(false);
            }
        }
        document.addEventListener("mousedown", handleClickOutside);
        return () => document.removeEventListener("mousedown", handleClickOutside);
    }, []);

    const handleNameChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const val = e.target.value;
        setMetricName(val);
        if (val.length > 1) {
            const sugs = await getSuggestions(val);
            setSuggestions(sugs);
            setShowSuggestions(true);
        } else {
            setSuggestions([]);
            setShowSuggestions(false);
        }
    };

    const selectSuggestion = (name: string) => {
        setMetricName(name);
        setShowSuggestions(false);
    };

    const addDimension = () => {
        setDimensions([...dimensions, { key: '', value: '' }]);
    };

    const updateDimension = (index: number, field: 'key' | 'value', val: string) => {
        const newDims = [...dimensions];
        newDims[index][field] = val;
        setDimensions(newDims);
    };

    const removeDimension = (index: number) => {
        setDimensions(dimensions.filter((_, i) => i !== index));
    };

    const handleSubmit = async (formData: FormData) => {
        // Construct dimensions object
        const dimsObj: Record<string, string> = {};
        dimensions.forEach(d => {
            if (d.key.trim()) dimsObj[d.key] = d.value;
        });

        formData.append('dimensions', JSON.stringify(dimsObj));

        await addEntry(formData);

        // Reset form
        setMetricName('');
        setValue('');
        setDimensions([]);
    };

    return (
        <motion.form
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            action={handleSubmit}
            className="glass p-6 rounded-3xl w-full max-w-lg mx-auto shadow-2xl relative z-10"
        >
            <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-cyan-400 mb-6 flex items-center gap-2">
                <Activity className="text-blue-400" /> Track Activity
            </h2>

            <div className="space-y-4">
                {/* Metric Name Input with Autocomplete */}
                <div className="relative" ref={wrapperRef}>
                    <input
                        name="metric_name"
                        type="text"
                        placeholder="Metric Name (e.g. Workout)"
                        value={metricName}
                        onChange={handleNameChange}
                        autoComplete="off"
                        required
                        className="input-premium w-full text-lg"
                    />
                    <AnimatePresence>
                        {showSuggestions && suggestions.length > 0 && (
                            <motion.ul
                                initial={{ opacity: 0, y: -10 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, y: -10 }}
                                className="absolute w-full bg-card/95 backdrop-blur-xl border border-white/10 mt-2 rounded-xl overflow-hidden z-50 max-h-40 overflow-y-auto shadow-xl"
                            >
                                {suggestions.map((s, i) => (
                                    <li
                                        key={i}
                                        onClick={() => selectSuggestion(s)}
                                        className="px-4 py-3 hover:bg-white/10 cursor-pointer text-sm text-gray-300 transition-colors"
                                    >
                                        {s}
                                    </li>
                                ))}
                            </motion.ul>
                        )}
                    </AnimatePresence>
                </div>

                {/* Value Input */}
                <div>
                    <input
                        name="value"
                        type="number"
                        step="any"
                        placeholder="Value (e.g. 1)"
                        value={value}
                        onChange={(e) => setValue(e.target.value)}
                        required
                        className="input-premium w-full text-lg font-mono"
                    />
                </div>

                {/* Dimensions */}
                <div className="space-y-3 pt-2">
                    <div className="flex justify-between items-center px-1">
                        <span className="text-sm font-medium text-muted-foreground flex items-center gap-1">
                            <Tag size={14} /> Dimensions
                        </span>
                        <button
                            type="button"
                            onClick={addDimension}
                            className="text-xs bg-secondary hover:bg-secondary/80 text-white px-3 py-1 rounded-full transition-colors"
                        >
                            + Add Tag
                        </button>
                    </div>

                    <AnimatePresence>
                        {dimensions.map((dim, i) => (
                            <motion.div
                                key={i}
                                initial={{ opacity: 0, height: 0 }}
                                animate={{ opacity: 1, height: 'auto' }}
                                exit={{ opacity: 0, height: 0 }}
                                className="flex gap-2"
                            >
                                <input
                                    placeholder="Key (e.g. type)"
                                    value={dim.key}
                                    onChange={(e) => updateDimension(i, 'key', e.target.value)}
                                    className="input-premium flex-1 text-sm py-2"
                                />
                                <input
                                    placeholder="Value (e.g. squats)"
                                    value={dim.value}
                                    onChange={(e) => updateDimension(i, 'value', e.target.value)}
                                    className="input-premium flex-1 text-sm py-2"
                                />
                                <button
                                    type="button"
                                    onClick={() => removeDimension(i)}
                                    className="text-destructive hover:bg-destructive/10 p-2 rounded-lg transition-colors"
                                >
                                    <X size={16} />
                                </button>
                            </motion.div>
                        ))}
                    </AnimatePresence>
                </div>

                <button
                    type="submit"
                    className="btn-premium w-full mt-6 flex justify-center items-center gap-2 group"
                >
                    <Plus className="group-hover:rotate-90 transition-transform duration-300" /> Log Entry
                </button>
            </div>
        </motion.form>
    );
}
