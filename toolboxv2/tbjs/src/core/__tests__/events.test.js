import EventBus from '../events.js';
import Logger from '../logger.js'; // Wird von EventBus intern genutzt

// Mock Logger, da EventBus ihn importiert
jest.mock('../logger.js', () => ({
  debug: jest.fn(),
  error: jest.fn(),
}));

describe('EventBus', () => {
  beforeEach(() => {
    // Reset listeners before each test
    EventBus._listeners = {};
    Logger.debug.mockClear();
    Logger.error.mockClear();
  });

  it('should add a listener and emit an event', () => {
    const mockCallback = jest.fn();
    EventBus.on('testEvent', mockCallback);
    EventBus.emit('testEvent', { data: 'payload' });

    expect(mockCallback).toHaveBeenCalledTimes(1);
    expect(mockCallback).toHaveBeenCalledWith({ data: 'payload' });
  });

  it('should allow multiple listeners for the same event', () => {
    const callback1 = jest.fn();
    const callback2 = jest.fn();
    EventBus.on('multiEvent', callback1);
    EventBus.on('multiEvent', callback2);
    EventBus.emit('multiEvent', 'data');

    expect(callback1).toHaveBeenCalledWith('data');
    expect(callback2).toHaveBeenCalledWith('data');
  });

  it('should remove a specific listener', () => {
    const callbackToKeep = jest.fn();
    const callbackToRemove = jest.fn();
    EventBus.on('removeTest', callbackToKeep);
    EventBus.on('removeTest', callbackToRemove);

    EventBus.off('removeTest', callbackToRemove);
    EventBus.emit('removeTest', 'testData');

    expect(callbackToKeep).toHaveBeenCalledWith('testData');
    expect(callbackToRemove).not.toHaveBeenCalled();
  });

  it('should do nothing if trying to remove a non-existent listener or event', () => {
    const callback = jest.fn();
    EventBus.on('exists', callback)
    expect(() => EventBus.off('nonExistentEvent', () => {})).not.toThrow();
    expect(() => EventBus.off('exists', () => {})).not.toThrow(); // Removing a listener not actually there
    EventBus.emit('exists', 'data');
    expect(callback).toHaveBeenCalled(); // Original listener should still be there
  });

  it('should handle emitting events with no listeners', () => {
    expect(() => EventBus.emit('noListenersEvent', 'data')).not.toThrow();
  });

  it('should call "once" listeners only once', () => {
    const onceCallback = jest.fn();
    EventBus.once('onceEvent', onceCallback);

    EventBus.emit('onceEvent', 'first');
    EventBus.emit('onceEvent', 'second');

    expect(onceCallback).toHaveBeenCalledTimes(1);
    expect(onceCallback).toHaveBeenCalledWith('first');
  });

  it('should log an error if a listener throws an error and continue calling other listeners', () => {
    const errorCallback = jest.fn(() => {
      throw new Error('Test listener error');
    });
    const normalCallback = jest.fn();

    EventBus.on('errorTest', errorCallback);
    EventBus.on('errorTest', normalCallback);

    EventBus.emit('errorTest', 'data');

    expect(errorCallback).toHaveBeenCalledWith('data');
    expect(Logger.error).toHaveBeenCalledWith(
      '[Events] Error in listener for errorTest:',
      expect.any(Error)
    );
    expect(normalCallback).toHaveBeenCalledWith('data');
  });
});
